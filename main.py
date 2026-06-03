"""
Orbit Wars - Adaptive Campaign + Logistics Guard Agent (v17)

Strategy: keep v16's tactical scoring, defense, wave attacks, and logistics guard,
but add campaign-level choices inspired by stronger public agents: stop expanding
when neutral ROI is low, avoid captures that are easy to snipe back, concentrate
multi-planet hammer attacks, and regroup surplus ships toward threatened frontiers.

Key design decisions (v17 changes from v16):
  NEW A. Strategic stop-expand mode suppresses low-value neutral captures once
         production share is healthy, the clock is past expansion tempo, or enemy
         launch tempo says the game has moved into combat.
  NEW B. Anti-snipe neutral guard rejects captures whose landing surplus is likely
         to be re-flipped by nearby enemy garrisons within a short response window.
  NEW C. Hammer attack coordinates up to three sources into one high-value enemy
         planet during combat/leader-pressure phases.
  NEW D. Regroup moves surplus rear ships toward owned planets under nearby enemy
         pressure when no urgent attack consumes them.

Retained from v16:
  - Adaptive modes: TURTLE, BALANCED, AGGRESSIVE, FINAL_PUSH
  - Wave attack and consolidation for enemy captures
  - Empty-city opportunity detection for drained enemy planets
  - TURTLE frontier filtering
  - 2-turn idle timeout when logistics are healthy
  - Friendly fleets count toward strategic strength totals
  - Logistics guard for midgame transit overextension
  - stop_leader_bonus, retake_penalty, dogpile_bonus, snipe_bonus
  - dominant_mode all-in, static_bonus=2.5, production^1.3 scoring
  - Phase 4 chip attacks on neutrals when logistics are healthy

Public API
----------
    agent(obs) -> list of [from_planet_id, angle_radians, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT

SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0

_turn            = 0
_turns_no_attack = 0   # consecutive turns with zero enemy-planet dispatches


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from v14)
# ---------------------------------------------------------------------------

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
    r     = math.hypot(planet.x - SUN_X, planet.y - SUN_Y)
    theta = math.atan2(planet.y - SUN_Y, planet.x - SUN_X) + ang_vel * turns
    return SUN_X + r * math.cos(theta), SUN_Y + r * math.sin(theta)


def _intercept(sx, sy, planet, ships, ang_vel, iters=5):
    """Iterative intercept: return (eta_turns, target_x, target_y)."""
    spd = _speed(max(1, ships))
    eta = math.hypot(planet.x - sx, planet.y - sy) / spd
    for _ in range(iters):
        tx, ty = _predict_pos(planet, eta, ang_vel)
        eta    = math.hypot(tx - sx, ty - sy) / spd
    tx, ty = _predict_pos(planet, eta, ang_vel)
    return eta, tx, ty


def _hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_R
    b    = 2.0 * (fx*dx + fy*dy)
    c    = fx*fx + fy*fy - SUN_R * SUN_R
    disc = b*b - 4.0*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 < t1 < 1.0) or (0.0 < t2 < 1.0) or (t1 <= 0.0 and t2 >= 1.0)


def _reserve(ships, turn):
    """Base reserve curve (unchanged from v14)."""
    if turn < 50:
        return max(2, int(ships * 0.04))
    if turn < 150:
        return max(10, int(ships * 0.18))
    if turn < 350:
        r = max(15, int(ships * 0.22))
    else:
        r = max(20, int(ships * 0.28))
    if turn > 400:
        r = max(10, int(r * 0.70))
    return r


# ---------------------------------------------------------------------------
# NEW v15: adaptive mode helpers
# ---------------------------------------------------------------------------

def _assess_mode(my_total, my_prod, enemy_totals, enemy_prods, turn):
    """Determine strategic mode for this turn based on relative strength."""
    best_enemy_total = max(enemy_totals, default=0)
    best_enemy_prod  = max(enemy_prods,  default=0)
    if turn > 450:
        return 'FINAL_PUSH'
    # Comfortably ahead on both ships and production ??turtle up and consolidate
    if my_total > best_enemy_total * 1.4 and my_prod > best_enemy_prod * 1.1:
        return 'TURTLE'
    # Meaningfully behind on ships OR outproduced late-game ??be aggressive
    if my_total < best_enemy_total * 0.75 or (my_prod < best_enemy_prod * 0.7 and turn > 100):
        return 'AGGRESSIVE'
    return 'BALANCED'


def _get_reserve(ships, turn, mode, elim=False, dominant=False,
                 four_p=False, behind=False):
    """Mode-aware reserve policy layered on top of v14 sub-conditions."""
    if mode == 'FINAL_PUSH':
        return 0
    base = _reserve(ships, turn)
    # v14 sub-conditions applied first (order matters: most aggressive first)
    if elim or dominant:
        base = base // 3
    elif four_p and turn < 60:
        base = base // 3
    elif behind:
        base = base // 2
    # Mode multiplier on top
    if mode == 'TURTLE':
        return int(base * 1.5)
    if mode == 'AGGRESSIVE':
        return max(0, base // 2)
    return base  # BALANCED


def _enemy_retake_risk(target, arrival_turn, landing_surplus, enemy_planets):
    """Approximate whether a neutral capture is easy for enemies to re-flip."""
    if target.owner != -1 or arrival_turn <= 0:
        return False
    for ep in enemy_planets:
        force = int(ep.ships * 0.45)
        if force < 8:
            continue
        d = math.hypot(ep.x - target.x, ep.y - target.y)
        eta = d / _speed(force)
        delay = eta - arrival_turn
        if delay < 1 or delay > 22:
            continue
        defender = landing_surplus + target.production * delay
        if force > defender + 2:
            return True
    return False


def _enemy_pressure_at(planet, enemy_planets, horizon=35.0):
    """Distance-decayed reachable enemy mass around an owned planet."""
    pressure = 0.0
    for ep in enemy_planets:
        ships = max(0, ep.ships)
        if ships < 5:
            continue
        reach = _speed(max(1, int(ships * 0.45))) * horizon
        d = math.hypot(ep.x - planet.x, ep.y - planet.y)
        if d >= reach:
            continue
        pressure += ships * (1.0 - d / max(reach, 1.0))
    return pressure


# ---------------------------------------------------------------------------
# Fleet parsing (unchanged from v14)
# ---------------------------------------------------------------------------

def _parse_fleets(fleets, player, pid_map, ang_vel):
    """Return (friendly_en_route, hostile_en_route, third_party_en_route).

    friendly:    our fleets  -> planet_id -> ships
    hostile:     all non-us  -> planet_id -> ships
    third_party: fleets by enemy X heading to enemy Y's planet (X?, X?layer)
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
            bearing        = math.atan2(pred_y - f.y, pred_x - f.x)
            diff           = abs(math.atan2(math.sin(f.angle - bearing),
                                            math.cos(f.angle - bearing)))
            dist_to_pred   = math.hypot(pred_x - f.x, pred_y - f.y)
            if diff < best_diff and dist_to_pred < 160:
                best_diff = diff
                best_pid  = pid
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def agent(obs):
    global _turn
    _turn += 1
    try:
        return _decide(obs)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def _decide(obs):
    global _turns_no_attack

    player  = _g(obs, "player", 0)
    ang_vel = _g(obs, "angular_velocity", 0.0)
    planets = [Planet(*p) for p in _g(obs, "planets", [])]
    fleets  = [Fleet(*f)  for f in _g(obs, "fleets",  [])]

    comet_ids = set(_g(obs, "comet_planet_ids", []))

    my_planets    = [p for p in planets if p.owner == player]
    all_targets   = [p for p in planets if p.owner != player]
    pid_map       = {p.id: p for p in planets}

    if not my_planets or not all_targets:
        return []

    my_prod       = sum(p.production for p in my_planets)
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    enemy_pids    = set(p.owner for p in enemy_planets)

    # ---- Strength totals ------------------------------------------------
    # Count our fleets too; otherwise large launches make us look falsely weak
    # and can trigger a self-reinforcing AGGRESSIVE over-extension loop.
    my_planet_ships = sum(p.ships for p in my_planets)
    my_fleet_ships  = sum(f.ships for f in fleets if f.owner == player)
    my_total        = my_planet_ships + my_fleet_ships
    avg_garrison    = my_planet_ships / max(len(my_planets), 1)
    transit_pressure = my_fleet_ships / max(my_planet_ships, 1)
    enemy_totals  = [
        sum(p.ships for p in enemy_planets if p.owner == e)
        + sum(f.ships for f in fleets if f.owner == e)
        for e in enemy_pids
    ]
    enemy_prods   = [
        sum(p.production for p in enemy_planets if p.owner == e)
        for e in enemy_pids
    ]

    # ---- Adaptive mode --------------------------------------------------
    mode = _assess_mode(my_total, my_prod, enemy_totals, enemy_prods, _turn)
    logistics_guard = (
        _turn < 160
        and mode != 'FINAL_PUSH'
        and my_fleet_ships > 80
        and transit_pressure > 1.6
        and avg_garrison < 18
    )

    max_enemy_prod    = max(enemy_prods, default=0)
    losing_prod       = max_enemy_prod > my_prod * 1.2

    max_enemy_planet_count = max(
        (sum(1 for p in enemy_planets if p.owner == e) for e in enemy_pids),
        default=0
    ) if enemy_pids else 0
    behind_on_planets = max_enemy_planet_count > len(my_planets) + 1

    neutrals_left = sum(1 for p in planets if p.owner == -1 and p.id not in comet_ids)
    few_neutrals  = neutrals_left < 4
    two_player    = len(enemy_pids) <= 1
    total_owned_prod = my_prod + sum(enemy_prods)
    my_prod_share = my_prod / max(total_owned_prod, 1)
    enemy_launch_sources = {
        getattr(f, 'from_planet_id', None)
        for f in fleets
        if f.owner not in (-1, player) and f.ships >= 20
    }
    enemy_tempo = len(enemy_launch_sources) >= 2
    stop_neutral_expand = (
        mode != 'FINAL_PUSH'
        and not losing_prod
        and not two_player
        and (
            (_turn >= 90 and (not two_player))
            or ((not two_player) and _turn >= 25 and my_prod_share >= 0.35)
            or ((not two_player) and enemy_tempo and _turn >= 35)
        )
    )

    # Weakest enemy (by combined ships + fleets) -------------------------
    enemy_strength = {}
    for p in enemy_planets:
        enemy_strength[p.owner] = enemy_strength.get(p.owner, 0.0) + p.ships
    for f in fleets:
        if f.owner not in (-1, player):
            enemy_strength[f.owner] = enemy_strength.get(f.owner, 0.0) + f.ships
    weakest_enemy = (min(enemy_strength, key=enemy_strength.get)
                     if enemy_strength else None)

    weakest_ships    = enemy_strength.get(weakest_enemy, 9999) if weakest_enemy else 9999
    # v13-FIX2: 40%/300 thresholds (reverted from v12's 50%/400)
    elimination_mode = (weakest_ships < my_total * 0.40 and weakest_ships < 300)

    dominant_mode = (
        len(enemy_pids) > 0
        and my_total > weakest_ships * 3.0
        and my_prod >= max_enemy_prod * 1.2
        and not elimination_mode
    )

    # Production leader (4-player pressure) ------------------------------
    enemy_prod_by_pid = {e: sum(p.production for p in enemy_planets if p.owner == e)
                         for e in enemy_pids}
    production_leader = (max(enemy_prod_by_pid, key=enemy_prod_by_pid.get)
                         if enemy_prod_by_pid else None)
    prod_leader_prod  = enemy_prod_by_pid.get(production_leader, 0)
    leader_is_threat  = len(enemy_pids) > 1 and prod_leader_prod > my_prod * 1.2
    combat_campaign = (
        few_neutrals or stop_neutral_expand or losing_prod
        or leader_is_threat or elimination_mode or dominant_mode
    )

    four_player      = len(enemy_pids) >= 2
    focused_enemy_4p = weakest_enemy if (four_player and weakest_enemy) else None

    friendly_en, hostile_en, third_party_en = _parse_fleets(
        fleets, player, pid_map, ang_vel)

    # ---- Idle timeout ---------------------------------------------------
    idle_boost = (_turns_no_attack >= 2)

    # ---- Empty-city opportunity detection (NEW v15) ---------------------
    # Detect recently drained enemy garrisons for 10x bonus targets.
    # Source 1: enemy planet garrison < 15 with no friendly already inbound.
    # Source 2: large hostile fleet (>30) just left that planet AND garrison low.
    fleet_drained = {}   # enemy_planet_id -> ships in departing hostile fleets
    for f in fleets:
        if f.owner not in (-1, player):
            from_pid = getattr(f, 'from_planet_id', None)
            if from_pid is not None:
                src = pid_map.get(from_pid)
                if src and src.owner == f.owner and f.ships > 30:
                    fleet_drained[from_pid] = fleet_drained.get(from_pid, 0) + f.ships

    empty_city_ids = set()
    for ep in enemy_planets:
        if friendly_en.get(ep.id, 0) > 0:
            continue   # we're already sending there
        if ep.ships < 15:
            empty_city_ids.add(ep.id)
        elif fleet_drained.get(ep.id, 0) > 30 and ep.ships < ep.production * 5:
            empty_city_ids.add(ep.id)

    # ---- TURTLE frontier target filtering (NEW v15) ---------------------
    # In TURTLE mode only attack planets within distance 25, top-3 by prod/dist.
    if mode == 'TURTLE':
        frontier = []
        for t in all_targets:
            min_d = min(math.hypot(t.x - m.x, t.y - m.y) for m in my_planets)
            if min_d <= 25:
                frontier.append((t, min_d))
        frontier.sort(key=lambda td: td[0].production / max(td[1], 1.0), reverse=True)
        targets = [td[0] for td in frontier[:3]] if frontier else all_targets[:3]
    else:
        targets = all_targets

    # ====================================================================
    moves       = []
    source_used = set()
    sending_to  = {}

    # ---- Phase 1: Reinforce threatened planets (unchanged from v14) -----
    for mine in sorted(my_planets, key=lambda p: -hostile_en.get(p.id, 0)):
        threat = hostile_en.get(mine.id, 0)
        if threat == 0:
            continue
        defense = mine.ships + friendly_en.get(mine.id, 0)
        if threat <= defense:
            continue
        shortfall   = int(threat - defense) + 2
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
            angle   = math.atan2(mine.y - ally.y, mine.x - ally.x)
            moves.append([ally.id, angle, contrib])
            source_used.add(ally.id)
            shortfall -= contrib
        # Lock defended planet too ??prevent Phase 3 from depleting it
        if shortfall < int(threat - defense) + 2:
            source_used.add(mine.id)

    # ---- Phase 2: Score all (source, target) pairs ----------------------
    candidates = []
    for mine in my_planets:
        if mine.id in source_used:
            continue
        base_res = _get_reserve(mine.ships, _turn, mode,
                                elim=elimination_mode, dominant=dominant_mode,
                                four_p=four_player, behind=behind_on_planets)
        avail = max(0, mine.ships - base_res)
        if avail < 1:
            continue

        for t in targets:
            eta, tx, ty = _intercept(mine.x, mine.y, t, avail, ang_vel)
            if _hits_sun(mine.x, mine.y, tx, ty):
                continue

            dist     = max(math.hypot(tx - mine.x, ty - mine.y), 1e-6)
            raw_garr = t.ships if t.owner == -1 else t.ships + t.production * eta
            already  = friendly_en.get(t.id, 0) + sending_to.get(t.id, 0)
            if t.owner not in (-1, player):
                already += third_party_en.get(t.id, 0)
            net_garr = max(0.0, raw_garr - already)

            static_bonus  = 2.5 if _is_static(t) else 1.0
            neutral_bonus = 1.5 if t.owner == -1 else 1.0
            if t.owner == -1 and t.id not in comet_ids and stop_neutral_expand:
                if t.production >= 4 and dist <= 22 and net_garr <= 22:
                    neutral_bonus = 0.8
                else:
                    neutral_bonus = 0.25
            comet_bonus   = 2.0 if t.id in comet_ids else 1.0

            # v13-FIX1: enemy_bonus 2.0 when neutrals?? (not 1.5)
            if t.owner not in (-1, player):
                if neutrals_left == 0:
                    enemy_bonus = 5.0
                elif few_neutrals:
                    enemy_bonus = 3.0
                else:
                    enemy_bonus = 2.0
                # NEW v15: AGGRESSIVE mode adds extra enemy incentive
                if mode == 'AGGRESSIVE':
                    enemy_bonus += 1.0
            else:
                enemy_bonus = 1.0

            prod_bonus = 2.0 if (t.owner not in (-1, player) and losing_prod) else 1.0
            elim_bonus = (5.0 if (dominant_mode   and t.owner == weakest_enemy) else
                          3.0 if (elimination_mode and t.owner == weakest_enemy) else
                          1.0)
            weak_bonus = (1.4 if (not elimination_mode and not dominant_mode
                                  and t.owner == weakest_enemy) else 1.0)

            third_ships   = third_party_en.get(t.id, 0)
            dogpile_bonus = 2.5 if third_ships > 0 else 1.0
            snipe_bonus   = (2.0 if (third_ships > 0 and t.owner not in (-1, player)
                                     and third_ships > t.ships * 0.5) else 1.0)

            # retake_penalty: penalise deep enemy clusters
            retake_penalty = 1.0
            if t.owner not in (-1, player):
                min_same_dist = min(
                    (math.hypot(ep.x - t.x, ep.y - t.y)
                     for ep in enemy_planets if ep.id != t.id and ep.owner == t.owner),
                    default=100.0
                )
                retake_penalty = max(0.4, min(1.0, min_same_dist / 30))

            stop_leader_bonus = (2.0 if (leader_is_threat and t.owner == production_leader
                                         and not elimination_mode and not dominant_mode)
                                 else 1.0)

            # v13-FIX3: focus-fire 3x weakest / 0.75x others
            if four_player and focused_enemy_4p is not None:
                if t.owner == focused_enemy_4p:
                    focus_4p = 3.0
                elif t.owner not in (-1, player):
                    focus_4p = 0.75
                else:
                    focus_4p = 1.0
            else:
                focus_4p = 1.0

            # NEW v15: empty-city opportunity ??10? bonus
            empty_city_bonus = 10.0 if t.id in empty_city_ids else 1.0

            score = (static_bonus * neutral_bonus * comet_bonus * enemy_bonus
                     * prod_bonus * elim_bonus * weak_bonus * dogpile_bonus * snipe_bonus
                     * retake_penalty * stop_leader_bonus * focus_4p * empty_city_bonus
                     * (t.production ** 1.3) / (dist * max(net_garr, 1.0)))
            candidates.append((score, mine.id, t.id, eta, tx, ty, net_garr, retake_penalty))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Build mine_avail dict using mode-aware reserves
    mine_avail = {}
    for p in my_planets:
        if p.id in source_used:
            continue
        res = _get_reserve(p.ships, _turn, mode,
                           elim=elimination_mode, dominant=dominant_mode,
                           four_p=four_player, behind=behind_on_planets)
        mine_avail[p.id] = max(0, p.ships - res)

    if logistics_guard:
        for pid in list(mine_avail):
            mine_avail[pid] = int(mine_avail[pid] * 0.45)

    # ---- v17 Hammer: one coordinated high-value enemy strike -------------
    hammer_fired = False
    hammer_ready = (not two_player) and _turn >= 60
    if hammer_ready and combat_campaign and not logistics_guard and enemy_planets:
        hammer_options = []
        for t in enemy_planets:
            if t.id in sending_to:
                continue
            if t.production < 2 and t.id not in empty_city_ids:
                continue
            contributors = []
            total = 0
            max_eta = 0.0
            for p in sorted(my_planets, key=lambda m: math.hypot(m.x - t.x, m.y - t.y)):
                if p.id in source_used:
                    continue
                avail = mine_avail.get(p.id, 0)
                if avail < 12:
                    continue
                sendable = min(avail, max(12, int(avail * 0.70)))
                eta_h, tx_h, ty_h = _intercept(p.x, p.y, t, sendable, ang_vel)
                if eta_h > 30 or _hits_sun(p.x, p.y, tx_h, ty_h):
                    continue
                contributors.append((eta_h, p, sendable))
                total += sendable
                max_eta = max(max_eta, eta_h)
                if len(contributors) >= 3:
                    break
            if len(contributors) < 2:
                continue
            already = friendly_en.get(t.id, 0) + sending_to.get(t.id, 0) + third_party_en.get(t.id, 0)
            needed = max(1, int(t.ships + t.production * max_eta - already))
            overkill = 1.18 if two_player else 1.28
            required = int(needed * overkill) + max(8, t.production * 4)
            if total < required:
                continue
            owner_bonus = (
                2.5 if t.owner == weakest_enemy else
                2.0 if leader_is_threat and t.owner == production_leader else
                1.0
            )
            empty_bonus = 3.0 if t.id in empty_city_ids else 1.0
            score = owner_bonus * empty_bonus * (t.production ** 1.5) / max(required * (1 + max_eta / 20), 1)
            hammer_options.append((score, t, contributors, required))

        if hammer_options:
            hammer_options.sort(key=lambda x: x[0], reverse=True)
            _, t, contributors, remaining = hammer_options[0]
            contributors.sort(key=lambda x: x[0])
            for _, p, sendable in contributors:
                if remaining <= 0:
                    break
                ships = min(sendable, max(12, remaining))
                eta_h, tx_h, ty_h = _intercept(p.x, p.y, t, ships, ang_vel)
                if _hits_sun(p.x, p.y, tx_h, ty_h):
                    continue
                angle = math.atan2(ty_h - p.y, tx_h - p.x)
                moves.append([p.id, angle, ships])
                mine_avail[p.id] = max(0, mine_avail.get(p.id, 0) - ships)
                source_used.add(p.id)
                sending_to[t.id] = sending_to.get(t.id, 0) + ships
                remaining -= ships
                hammer_fired = True

    # ---- Phase 3: Greedy dispatch + Wave 2 consolidation ----------------
    enemy_dispatched       = set()
    enemy_dispatched_count = 1 if hammer_fired else 0   # track for idle-timeout reset

    for score, mine_id, t_id, eta, tx, ty, _, retake_p in candidates:
        avail = mine_avail.get(mine_id, 0)
        if avail < 1:
            continue

        t          = pid_map[t_id]
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
            if stop_neutral_expand and not is_comet:
                if t.production < 4 or eta > 20 or net_garr > 22:
                    continue
            if avail >= needed:
                ships_to_send = needed
            elif avail > net_garr:
                ships_to_send = avail   # barely captures
            else:
                continue
        else:
            # v13-IMP3: adaptive prod_buffer
            prod_buffer = max(8, int(t.production * 6))
            if elimination_mode and t.owner == weakest_enemy:
                prod_buffer = max(4, int(t.production * 3))
            elif losing_prod and t.owner not in (-1, player):
                prod_buffer = max(6, int(t.production * 4))

            # NEW v15: idle timeout ??cut prod_buffer 30% if stalled
            if idle_boost and not logistics_guard:
                prod_buffer = int(prod_buffer * 0.70)

            needed = int(net_garr) + prod_buffer

            if avail >= needed:
                ships_to_send = needed
            elif (not logistics_guard) and idle_boost and avail >= int(needed * 0.70):
                # Idle timeout: accept 70% of needed to break stalemate
                ships_to_send = avail
            elif avail >= max(10, int(net_garr * 0.40)) and (
                    not logistics_guard) and (
                    losing_prod or few_neutrals or elimination_mode
                    or dominant_mode or behind_on_planets):
                ships_to_send = avail
            else:
                continue

        if ships_to_send < 1:
            continue
        mine = pid_map[mine_id]
        if two_player:
            eta_fire, tx_fire, ty_fire = eta, tx, ty
        else:
            eta_fire, tx_fire, ty_fire = _intercept(mine.x, mine.y, t, ships_to_send, ang_vel)
        if (not two_player) and is_neutral and not is_comet:
            surplus = ships_to_send - net_garr
            if _enemy_retake_risk(t, eta_fire, surplus, enemy_planets):
                extra = min(avail - ships_to_send, max(0, int(t.production * 5 + 8 - surplus)))
                if t.production >= 3 and extra > 0:
                    ships_to_send += extra
                    eta_fire, tx_fire, ty_fire = _intercept(mine.x, mine.y, t, ships_to_send, ang_vel)
                    surplus = ships_to_send - net_garr
                if _enemy_retake_risk(t, eta_fire, surplus, enemy_planets):
                    continue
        if _hits_sun(mine.x, mine.y, tx_fire, ty_fire):
            continue

        angle = math.atan2(ty_fire - mine.y, tx_fire - mine.x)
        moves.append([mine_id, angle, ships_to_send])
        sending_to[t_id]      = sending_to.get(t_id, 0) + ships_to_send
        mine_avail[mine_id]  -= ships_to_send
        if is_enemy:
            enemy_dispatched.add(mine_id)
            enemy_dispatched_count += 1

        # Lock planet after dispatch if reserves are getting thin
        base_res = _reserve(pid_map[mine_id].ships, _turn)
        if mine_avail.get(mine_id, 0) < base_res * 2:
            source_used.add(mine_id)

        # ---- NEW v15: Wave 2 consolidation (enemy targets only, not TURTLE) ----
        if two_player:
            allow_wave2 = is_enemy and mode != 'TURTLE' and not logistics_guard
        else:
            allow_wave2 = (
                is_enemy and mode != 'TURTLE' and not logistics_guard
                and (t.production >= 3 or t.id in empty_city_ids or losing_prod
                     or elimination_mode or dominant_mode)
                and retake_p >= 0.55
            )
        if allow_wave2:
            w2_candidates = []
            for p2 in my_planets:
                if p2.id == mine_id or p2.id in source_used:
                    continue
                p2_avail = mine_avail.get(p2.id, 0)
                if p2_avail < 10:
                    continue
                p2_eta, p2_tx, p2_ty = _intercept(p2.x, p2.y, t, p2_avail, ang_vel)
                if p2_eta > eta + 5:
                    continue   # arrives too late to consolidate
                if _hits_sun(p2.x, p2.y, p2_tx, p2_ty):
                    continue
                w2_candidates.append((p2_eta, p2, p2_avail, p2_tx, p2_ty))

            if w2_candidates:
                # Pick earliest-arriving secondary source
                w2_candidates.sort(key=lambda x: x[0])
                _, p2, p2_avail, p2_tx, p2_ty = w2_candidates[0]
                w2_frac = 0.40 if two_player else 0.30
                w2_ships = max(10, int(p2_avail * w2_frac))
                w2_angle = math.atan2(p2_ty - p2.y, p2_tx - p2.x)
                moves.append([p2.id, w2_angle, w2_ships])
                mine_avail[p2.id]    = max(0, mine_avail.get(p2.id, 0) - w2_ships)
                sending_to[t_id]     = sending_to.get(t_id, 0) + w2_ships
                # Reduce P2 availability; locking fully only if reserve is thin
                p2_base_res = _reserve(p2.ships, _turn)
                if mine_avail.get(p2.id, 0) < p2_base_res * 2:
                    source_used.add(p2.id)

    # ---- Phase 4: Chip attacks on neutrals (disabled in TURTLE / FINAL_PUSH) ----
    if mode not in ('TURTLE', 'FINAL_PUSH') and not logistics_guard and not stop_neutral_expand:
        chipped_this_turn = set()
        chip_targets = [p for p in planets if p.owner == -1
                        and p.id not in comet_ids and p.production >= 2]
        for mine in sorted([p for p in my_planets if p.id not in source_used],
                           key=lambda p: mine_avail.get(p.id, 0), reverse=True):
            avail = mine_avail.get(mine.id, 0)
            if avail < 25:
                continue
            if hostile_en.get(mine.id, 0) > 0:
                continue   # planet under threat ??keep ships
            for t in sorted(chip_targets,
                            key=lambda n: math.hypot(n.x - mine.x, n.y - mine.y)):
                already  = sending_to.get(t.id, 0)
                eff_garr = t.ships - already
                if eff_garr <= 0 or avail >= eff_garr + 2:
                    continue   # already capturing or can capture directly
                if t.id in chipped_this_turn:
                    continue   # another planet already chipping this one
                chip = min(avail - 10, max(15, eff_garr - avail * 2))
                if chip < 10:
                    continue
                eta_c, tx_c, ty_c = _intercept(mine.x, mine.y, t, chip, ang_vel)
                if _hits_sun(mine.x, mine.y, tx_c, ty_c):
                    continue
                angle = math.atan2(ty_c - mine.y, tx_c - mine.x)
                moves.append([mine.id, angle, chip])
                mine_avail[mine.id] = max(0, mine_avail.get(mine.id, 0) - chip)
                sending_to[t.id]    = sending_to.get(t.id, 0) + chip
                chipped_this_turn.add(t.id)
                break   # one chip target per source planet per turn

    # ---- Phase 5: Regroup surplus toward pressured frontiers --------------
    if not two_player and not logistics_guard and mode != 'FINAL_PUSH' and _turn > 25:
        pressure = {p.id: _enemy_pressure_at(p, enemy_planets) for p in my_planets}
        regroup_targets = sorted(
            [p for p in my_planets if pressure.get(p.id, 0.0) > 18 and p.ships < avg_garrison + 25],
            key=lambda p: (pressure.get(p.id, 0.0), -p.ships),
            reverse=True,
        )
        regroup_count = 0
        for target in regroup_targets:
            if regroup_count >= 2:
                break
            desired = max(0, int(avg_garrison + pressure[target.id] * 0.12 - target.ships))
            if desired < 12:
                continue
            sources = sorted(
                [p for p in my_planets
                 if p.id != target.id and p.id not in source_used
                 and mine_avail.get(p.id, 0) >= 35
                 and pressure.get(p.id, 0.0) + 5 < pressure.get(target.id, 0.0)],
                key=lambda p: math.hypot(p.x - target.x, p.y - target.y)
            )
            for src in sources:
                avail = mine_avail.get(src.id, 0)
                send = min(desired, max(12, int(avail * 0.25)), avail - 20)
                if send < 12:
                    continue
                eta_r, tx_r, ty_r = _intercept(src.x, src.y, target, send, ang_vel)
                if eta_r > 18 or _hits_sun(src.x, src.y, tx_r, ty_r):
                    continue
                moves.append([src.id, math.atan2(ty_r - src.y, tx_r - src.x), send])
                mine_avail[src.id] = max(0, mine_avail.get(src.id, 0) - send)
                source_used.add(src.id)
                regroup_count += 1
                break

    # ---- Update idle timeout counter ------------------------------------
    if enemy_dispatched_count > 0:
        _turns_no_attack = 0
    else:
        _turns_no_attack += 1

    return moves
