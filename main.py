"""
Orbit Wars - Tier 2 Efficient Expansion Agent  (v14)

Strategy: aggressive expansion with chip attacks, focus-fire WEAKEST enemy (by total
ships), eliminate players early in 4-player games. Dominant-mode all-in when winning.
Prefer frontier enemy planets (isolated, defensible) over deep-cluster targets.

Key design decisions (v14 changes from v13):
  FIX A. Removed inner_priority 1.3x boost (IMP 4 from v13). Inner orbiting planets
     are hard to predict and defend; the 1.3x multiplier was sending fleets to planets
     that enemies could recapture before we consolidated, wasting ships.
  FIX B. Reverted comet_bonus to flat 2.0 (removed v13's urgency 3.0 when eta<50).
     The 3x urgency over-committed ships to comets and left regular planets exposed.

Retained from v13 (regressions fixed + improvements):
  - FIX 1: enemy_bonus 2.0 (not 1.5) when neutrals_left ≥ 4
  - FIX 2: elim threshold 40%/300 (not 50%/400)
  - FIX 3: softened focus-fire 3x/0.75x (not 4x/0.5x)
  - IMP 1: weakest enemy focus-fire
  - IMP 3: adaptive prod_buffer (4/3x elim, 6/4x losing_prod)
  - stop_leader_bonus (1.2x threshold, 2x bonus)
  - 4-player early-game reserve//3 when turn < 60
  - retake_penalty (0.4–1.0 based on cluster distance)
  - dominant_mode all-in (3x ships + 1.2x prod)
  - Phase 4 chip attacks on neutrals
  - zero-neutral enemy_bonus 5x / few-neutral 3x

Public API
----------
    agent(obs) -> list of [from_planet_id, angle_radians, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT

SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0

_turn = 0


def _g(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def _is_static(planet):
    return math.hypot(planet.x - SUN_X, planet.y - SUN_Y) + planet.radius >= ROTATION_RADIUS_LIMIT


def _predict_pos(planet, turns, ang_vel):
    if _is_static(planet) or turns == 0:
        return planet.x, planet.y
    r = math.hypot(planet.x - SUN_X, planet.y - SUN_Y)
    theta = math.atan2(planet.y - SUN_Y, planet.x - SUN_X) + ang_vel * turns
    return SUN_X + r * math.cos(theta), SUN_Y + r * math.sin(theta)


def _intercept(sx, sy, planet, ships, ang_vel, iters=5):
    """Iterative intercept: return (eta_turns, target_x, target_y)."""
    spd = _speed(max(1, ships))
    eta = math.hypot(planet.x - sx, planet.y - sy) / spd
    for _ in range(iters):
        tx, ty = _predict_pos(planet, eta, ang_vel)
        eta = math.hypot(tx - sx, ty - sy) / spd
    tx, ty = _predict_pos(planet, eta, ang_vel)
    return eta, tx, ty


def _hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_R
    b = 2.0 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - SUN_R * SUN_R
    disc = b*b - 4.0*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 < t1 < 1.0) or (0.0 < t2 < 1.0) or (t1 <= 0.0 and t2 >= 1.0)


def _reserve(ships, turn):
    """Ships to keep at home. Aggressive early, moderate late, all-in endgame."""
    if turn < 50:
        return max(2, int(ships * 0.04))
    if turn < 150:
        return max(10, int(ships * 0.18))
    if turn < 350:
        r = max(15, int(ships * 0.22))
    else:
        r = max(20, int(ships * 0.28))
    # [v10] clock pressure: final 100 turns → shed 30% more reserve
    if turn > 400:
        r = max(10, int(r * 0.70))
    return r


def _parse_fleets(fleets, player, pid_map, ang_vel):
    """Return (friendly_en_route, hostile_en_route, third_party_en_route).

    friendly:    our fleets  -> planet_id -> ships
    hostile:     all non-us  -> planet_id -> ships
    third_party: fleets by enemy X heading to enemy Y's planet (X!=Y, X!=player)
    """
    friendly, hostile, third_party = {}, {}, {}
    for f in fleets:
        best_pid, best_diff = None, 0.30
        spd = _speed(max(1, f.ships))
        for pid, p in pid_map.items():
            rough_eta = math.hypot(p.x - f.x, p.y - f.y) / spd
            if rough_eta > 250:
                continue
            pred_x, pred_y = _predict_pos(p, rough_eta, ang_vel)
            bearing = math.atan2(pred_y - f.y, pred_x - f.x)
            diff = abs(math.atan2(math.sin(f.angle - bearing),
                                  math.cos(f.angle - bearing)))
            dist_to_pred = math.hypot(pred_x - f.x, pred_y - f.y)
            if diff < best_diff and dist_to_pred < 160:
                best_diff = diff
                best_pid = pid
        if best_pid is None:
            continue
        if f.owner == player:
            friendly[best_pid] = friendly.get(best_pid, 0) + f.ships
        else:
            hostile[best_pid] = hostile.get(best_pid, 0) + f.ships
            tgt = pid_map.get(best_pid)
            if tgt and tgt.owner not in (-1, player) and f.owner != tgt.owner:
                third_party[best_pid] = third_party.get(best_pid, 0) + f.ships
    return friendly, hostile, third_party


def agent(obs):
    global _turn
    _turn += 1
    try:
        return _decide(obs)
    except Exception:
        return []


def _decide(obs):
    player  = _g(obs, "player", 0)
    ang_vel = _g(obs, "angular_velocity", 0.0)
    planets = [Planet(*p) for p in _g(obs, "planets", [])]
    fleets  = [Fleet(*f)  for f in _g(obs, "fleets",  [])]

    comet_ids = set(_g(obs, "comet_planet_ids", []))

    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player]
    pid_map    = {p.id: p for p in planets}

    if not my_planets or not targets:
        return []

    my_prod    = sum(p.production for p in my_planets)
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    enemy_pids = set(p.owner for p in enemy_planets)

    max_enemy_prod = max(
        (sum(p.production for p in enemy_planets if p.owner == e) for e in enemy_pids),
        default=0
    ) if enemy_pids else 0
    losing_prod = max_enemy_prod > my_prod * 1.2

    max_enemy_planet_count = max(
        (sum(1 for p in enemy_planets if p.owner == e) for e in enemy_pids),
        default=0
    ) if enemy_pids else 0
    behind_on_planets = max_enemy_planet_count > len(my_planets) + 1
    neutrals_left = sum(1 for p in planets if p.owner == -1 and p.id not in comet_ids)
    few_neutrals = neutrals_left < 4

    enemy_strength = {}
    for p in enemy_planets:
        enemy_strength[p.owner] = enemy_strength.get(p.owner, 0.0) + p.ships
    for f in fleets:
        if f.owner not in (-1, player):
            enemy_strength[f.owner] = enemy_strength.get(f.owner, 0.0) + f.ships
    weakest_enemy = min(enemy_strength, key=enemy_strength.get) if enemy_strength else None

    my_total = sum(p.ships for p in my_planets)
    weakest_ships = enemy_strength.get(weakest_enemy, 9999) if weakest_enemy else 9999
    # [v13-FIX2] reverted to v6 values: 40% / 300 (v12's 50%/400 triggered too early)
    elimination_mode = (weakest_ships < my_total * 0.40 and weakest_ships < 300)

    # dominant_mode: we're clearly winning → go all-in to close the game
    dominant_mode = (
        len(enemy_pids) > 0
        and my_total > weakest_ships * 3.0
        and my_prod >= max_enemy_prod * 1.2
        and not elimination_mode  # don't double-apply
    )

    # [v11] production_leader: in 4-player, identify and pressure the biggest producer
    enemy_prod_by_pid = {e: sum(p.production for p in enemy_planets if p.owner == e)
                         for e in enemy_pids}
    production_leader = max(enemy_prod_by_pid, key=enemy_prod_by_pid.get) if enemy_prod_by_pid else None
    prod_leader_prod  = enemy_prod_by_pid.get(production_leader, 0)
    # Threat = leader outproducing us by 20%+ in a multi-enemy game  [v12] 1.5→1.2
    leader_is_threat  = len(enemy_pids) > 1 and prod_leader_prod > my_prod * 1.2

    # [v13-IMP1] 4-player focus-fire: concentrate on the WEAKEST enemy (by total ships).
    # Weakest = min total ships across all their planets + fleets.
    # That enemy gets 3x attack bonus; all others get 0.75x penalty (softened from 0.5x).
    four_player = len(enemy_pids) >= 2
    focused_enemy_4p = None
    if four_player and enemy_pids and weakest_enemy is not None:
        focused_enemy_4p = weakest_enemy

    friendly_en, hostile_en, third_party_en = _parse_fleets(fleets, player, pid_map, ang_vel)

    moves       = []
    source_used = set()
    sending_to  = {}

    # Phase 1: Reinforce threatened planets
    for mine in sorted(my_planets, key=lambda p: -hostile_en.get(p.id, 0)):
        threat = hostile_en.get(mine.id, 0)
        if threat == 0:
            continue
        defense = mine.ships + friendly_en.get(mine.id, 0)
        if threat <= defense:
            continue
        shortfall = int(threat - defense) + 2
        reserve_div = 3 if (elimination_mode or dominant_mode) else 2
        allies = sorted(
            [a for a in my_planets if a.id != mine.id and a.id not in source_used],
            key=lambda a: math.hypot(a.x - mine.x, a.y - mine.y)
        )
        for ally in allies:
            if shortfall <= 0:
                break
            sendable = max(0, ally.ships - _reserve(ally.ships, _turn) // reserve_div)
            if sendable < 1:
                continue
            contrib = min(sendable, shortfall + 5)
            angle = math.atan2(mine.y - ally.y, mine.x - ally.x)
            moves.append([ally.id, angle, contrib])
            source_used.add(ally.id)
            shortfall -= contrib
        # [v11] Lock the defended planet too — prevent Phase 3 from depleting it
        if shortfall < int(threat - defense) + 2:  # at least some help was sent
            source_used.add(mine.id)

    # Phase 2: Score all (source, target) pairs
    candidates = []
    for mine in my_planets:
        if mine.id in source_used:
            continue
        if (elimination_mode or dominant_mode) or (four_player and _turn < 60):
            base_res = _reserve(mine.ships, _turn) // 3  # [v12] 4p early-game: match starter 0%-reserve
        elif behind_on_planets:
            base_res = _reserve(mine.ships, _turn) // 2
        else:
            base_res = _reserve(mine.ships, _turn)
        avail = max(0, mine.ships - base_res)
        if avail < 1:
            continue

        for t in targets:
            eta, tx, ty = _intercept(mine.x, mine.y, t, avail, ang_vel)
            if _hits_sun(mine.x, mine.y, tx, ty):
                continue

            dist = max(math.hypot(tx - mine.x, ty - mine.y), 1e-6)

            raw_garr = t.ships if t.owner == -1 else t.ships + t.production * eta
            already  = friendly_en.get(t.id, 0) + sending_to.get(t.id, 0)
            if t.owner not in (-1, player):
                already += third_party_en.get(t.id, 0)
            net_garr = max(0.0, raw_garr - already)

            static_bonus  = 2.5 if _is_static(t) else 1.0
            neutral_bonus = 1.5 if t.owner == -1 else 1.0
            comet_bonus = 2.0 if t.id in comet_ids else 1.0
            # [v13-FIX1] reverted neutral-first: enemy_bonus stays 2.0 when neutrals≥4
            # (v11/v12 used 1.5 when neutrals≥6, making us too passive vs enemies early)
            if t.owner not in (-1, player):
                if neutrals_left == 0:
                    enemy_bonus = 5.0
                elif few_neutrals:   # neutrals_left < 4
                    enemy_bonus = 3.0
                else:
                    enemy_bonus = 2.0
            else:
                enemy_bonus = 1.0
            prod_bonus    = 2.0 if (t.owner not in (-1, player) and losing_prod) else 1.0  # [v10] 1.6→2.0
            elim_bonus    = (5.0 if (dominant_mode and t.owner == weakest_enemy)
                             else 3.0 if (elimination_mode and t.owner == weakest_enemy)
                             else 1.0)
            weak_bonus    = 1.4 if (not elimination_mode and not dominant_mode and t.owner == weakest_enemy) else 1.0
            # [v10] dogpile 1.8→2.5; snipe_bonus when third-party already > 50% of garrison
            third_ships   = third_party_en.get(t.id, 0)
            dogpile_bonus = 2.5 if third_ships > 0 else 1.0
            snipe_bonus   = 2.0 if (third_ships > 0 and t.owner not in (-1, player)
                                    and third_ships > t.ships * 0.5) else 1.0
            # [v11] retake penalty: how quickly can same-enemy planets reinforce this target?
            # High-risk targets (surrounded by same-enemy planets) get penalized
            retake_penalty = 1.0
            if t.owner not in (-1, player):
                min_same_enemy_dist = min(
                    (math.hypot(ep.x - t.x, ep.y - t.y)
                     for ep in enemy_planets if ep.id != t.id and ep.owner == t.owner),
                    default=100.0
                )
                # Penalty scales: 1.0 (safe, far) to 0.4 (risky, close cluster)
                retake_penalty = max(0.4, min(1.0, min_same_enemy_dist / 30))
            # [v11] stop_leader_bonus: in 4-player, pressure the production leader
            stop_leader_bonus = (2.0 if (leader_is_threat and t.owner == production_leader
                                         and not elimination_mode and not dominant_mode)
                                 else 1.0)
            # [v13-FIX3] softened focus-fire: 3x on weakest enemy (was 4x), 0.75x on others (was 0.5x)
            if four_player and focused_enemy_4p is not None:
                if t.owner == focused_enemy_4p:
                    focus_4p = 3.0   # concentrate on weakest — still strong bonus
                elif t.owner not in (-1, player):
                    focus_4p = 0.75  # soft penalty; don't completely ignore easy picks
                else:
                    focus_4p = 1.0
            else:
                focus_4p = 1.0
            score = (static_bonus * neutral_bonus * comet_bonus * enemy_bonus
                     * prod_bonus * elim_bonus * weak_bonus * dogpile_bonus * snipe_bonus
                     * retake_penalty * stop_leader_bonus * focus_4p
                     * (t.production ** 1.3) / (dist * max(net_garr, 1.0)))
            candidates.append((score, mine.id, t.id, eta, tx, ty, net_garr, retake_penalty))

    candidates.sort(key=lambda c: c[0], reverse=True)

    if (elimination_mode or dominant_mode) or (four_player and _turn < 60):
        mine_avail = {p.id: max(0, p.ships - _reserve(p.ships, _turn) // 3)
                      for p in my_planets if p.id not in source_used}  # [v12] 4p early: min reserve
    elif behind_on_planets:
        mine_avail = {p.id: max(0, p.ships - _reserve(p.ships, _turn) // 2)
                      for p in my_planets if p.id not in source_used}
    else:
        mine_avail = {p.id: max(0, p.ships - _reserve(p.ships, _turn))
                      for p in my_planets if p.id not in source_used}

    # Phase 3: Greedy multi-target dispatch
    # Multi-fleet is legal per env source. Neutrals: multi-dispatch allowed.
    # Enemies: one attack per planet per turn to prevent ship bleed.
    enemy_dispatched = set()
    for score, mine_id, t_id, eta, tx, ty, _, retake_p in candidates:
        avail = mine_avail.get(mine_id, 0)
        if avail < 1:
            continue

        t = pid_map[t_id]
        is_neutral = t.owner == -1
        is_comet   = t.id in comet_ids
        is_enemy   = t.owner not in (-1, player)

        if is_enemy and mine_id in enemy_dispatched:
            continue

        already = friendly_en.get(t_id, 0) + sending_to.get(t_id, 0)
        if is_neutral:
            net_garr = max(0.0, t.ships - already)
        else:
            net_garr = max(0.0, t.ships + t.production * eta - already
                           - third_party_en.get(t_id, 0))

        if is_neutral or is_comet:
            needed = int(net_garr) + 2
            if net_garr <= 0:
                continue
            if avail >= needed:
                ships_to_send = needed
            elif avail > net_garr:
                ships_to_send = avail  # barely captures
            else:
                continue
        else:
            # [v13-IMP3] adaptive prod_buffer: lower when elimination_mode or losing prod
            prod_buffer = max(8, int(t.production * 6))
            if elimination_mode and t.owner == weakest_enemy:
                prod_buffer = max(4, int(t.production * 3))   # aggressive vs dying enemy
            elif losing_prod and t.owner not in (-1, player):
                prod_buffer = max(6, int(t.production * 4))   # more willing to strike when behind
            needed = int(net_garr) + prod_buffer
            if avail >= needed:
                ships_to_send = needed
            elif avail >= max(10, int(net_garr * 0.40)) and (losing_prod or few_neutrals or elimination_mode or dominant_mode or behind_on_planets):  # v9b: restored guard + dominant_mode
                ships_to_send = avail
            else:
                continue

        if ships_to_send < 1:
            continue
        mine = pid_map[mine_id]
        if _hits_sun(mine.x, mine.y, tx, ty):
            continue

        angle = math.atan2(ty - mine.y, tx - mine.x)
        moves.append([mine_id, angle, ships_to_send])
        sending_to[t_id] = sending_to.get(t_id, 0) + ships_to_send
        mine_avail[mine_id] -= ships_to_send
        if is_enemy:
            enemy_dispatched.add(mine_id)
        # Lock planet after dispatch if reserves are getting thin
        # (prevents multi-fleet draining planets below safe defense level)
        base_res = _reserve(pid_map[mine_id].ships, _turn)
        if mine_avail.get(mine_id, 0) < base_res * 2:
            source_used.add(mine_id)

    # Phase 4: single-source chip attacks on neutrals we can't yet capture directly.
    # Neutrals don't regenerate — chips stack, enabling capture earlier.
    # Only ONE source per neutral per turn to prevent multi-chip ship waste.
    # Only chip when planet is NOT under threat and has large surplus.
    chipped_this_turn = set()
    chip_targets = [p for p in planets if p.owner == -1
                    and p.id not in comet_ids and p.production >= 2]
    for mine in sorted([p for p in my_planets if p.id not in source_used],
                       key=lambda p: mine_avail.get(p.id, 0), reverse=True):
        avail = mine_avail.get(mine.id, 0)
        if avail < 25:
            continue
        if hostile_en.get(mine.id, 0) > 0:
            continue  # planet is under threat — keep ships for defense
        for t in sorted(chip_targets,
                        key=lambda n: math.hypot(n.x - mine.x, n.y - mine.y)):
            already = sending_to.get(t.id, 0)
            eff_garr = t.ships - already
            if eff_garr <= 0 or avail >= eff_garr + 2:
                continue  # already being captured or we can capture directly
            if t.id in chipped_this_turn:
                continue  # another planet is already chipping this neutral
            # Reduce garrison to ~2x current avail so we can capture when ships double
            chip = min(avail - 10, max(15, eff_garr - avail * 2))
            if chip < 10:
                continue
            eta, tx, ty = _intercept(mine.x, mine.y, t, chip, ang_vel)
            if _hits_sun(mine.x, mine.y, tx, ty):
                continue
            angle = math.atan2(ty - mine.y, tx - mine.x)
            moves.append([mine.id, angle, chip])
            mine_avail[mine.id] = max(0, mine_avail.get(mine.id, 0) - chip)
            sending_to[t.id] = sending_to.get(t.id, 0) + chip
            chipped_this_turn.add(t.id)
            break  # one chip target per source planet per turn

    return moves