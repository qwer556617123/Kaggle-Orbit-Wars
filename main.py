"""
Orbit Wars — Tier 2 Efficient Expansion Agent  (v3 — 4-player aware)

Strategy: expand precisely without wasting ships, then switch to harassment/capture
in direct combat when few neutrals remain.

Key design decisions:
  1. No chip attacks on neutrals — only send when we can guarantee capture.
  2. Multiple source planets can target the same neutral cooperatively.
  3. Adaptive reserve: 8% early → 30% mid → 35% late, with halved reserve when
     behind on planet count (aggressive catch-up mode).
  4. Orbit-aware fleet detection: predict planet positions at fleet ETA to correctly
     identify incoming threat targets for both orbiting and static planets.
  5. Multi-ally reinforcement in Phase 1: cover incoming fleet threats with
     combined fleets from multiple nearby planets.
  6. Direct combat mode (< 4 neutrals): harass enemy garrisons even without
     guarantee of capture, scoring enemy planets 3× higher than neutral.
  7. [v3] 4-player: proper third-party dogpile detection — distinguish enemy
     reinforcements from a different enemy attacking the same planet.
     Subtract confirmed third-party attackers from garrison estimate.
  8. [v3] 4-player: 1.4x bonus for targeting the weakest enemy player's planets.
  9. [v3] 4-player: 1.5x dogpile bonus when another enemy fleet is already
     attacking the target (based on third-party fleet detection).

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
    """Ships to keep at home."""
    if turn < 60:
        return max(4, int(ships * 0.08))
    if turn < 150:
        return max(16, int(ships * 0.30))
    if turn < 350:
        return max(20, int(ships * 0.35))
    return max(24, int(ships * 0.40))


def _parse_fleets(fleets, player, pid_map, ang_vel):
    """Return (friendly_en_route, hostile_en_route, third_party_en_route).

    friendly:    our fleets  → planet_id -> ships
    hostile:     all non-us  → planet_id -> ships
    third_party: fleets by enemy X heading to enemy Y's planet (X≠Y, X≠player)
                 — real dogpile signal usable in 4-player games
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
            # Third-party: attacker ≠ planet owner (genuine dogpile, not reinforcement)
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

    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player]
    pid_map    = {p.id: p for p in planets}

    if not my_planets or not targets:
        return []

    my_prod    = sum(p.production for p in my_planets)
    their_prod = sum(p.production for p in planets if p.owner not in (-1, player))
    losing_prod = their_prod > my_prod * 1.2

    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    behind_on_planets = len(enemy_planets) > len(my_planets) + 1
    neutrals_left = sum(1 for p in planets if p.owner == -1)
    few_neutrals = neutrals_left < 4

    # ── 4-player: track each enemy's total strength (planets + fleets) ────────
    enemy_strength: dict[int, float] = {}
    for p in enemy_planets:
        enemy_strength[p.owner] = enemy_strength.get(p.owner, 0.0) + p.ships
    for f in fleets:
        if f.owner not in (-1, player):
            enemy_strength[f.owner] = enemy_strength.get(f.owner, 0.0) + f.ships
    weakest_enemy = min(enemy_strength, key=enemy_strength.get) if enemy_strength else None

    friendly_en, hostile_en, third_party_en = _parse_fleets(fleets, player, pid_map, ang_vel)

    moves       = []
    source_used = set()
    sending_to  = {}   # target_id -> ships we dispatch this turn

    # ── Phase 1: Reinforce threatened planets ─────────────────────────────────
    for mine in sorted(my_planets, key=lambda p: -hostile_en.get(p.id, 0)):
        threat = hostile_en.get(mine.id, 0)
        if threat == 0:
            continue
        defense = mine.ships + friendly_en.get(mine.id, 0)
        if threat <= defense:
            continue
        shortfall = int(threat - defense) + 2
        allies = sorted(
            [a for a in my_planets if a.id != mine.id and a.id not in source_used],
            key=lambda a: math.hypot(a.x - mine.x, a.y - mine.y)
        )
        for ally in allies:
            if shortfall <= 0:
                break
            sendable = max(0, ally.ships - _reserve(ally.ships, _turn) // 2)
            if sendable < 1:
                continue
            contrib = min(sendable, shortfall + 5)
            angle = math.atan2(mine.y - ally.y, mine.x - ally.x)
            moves.append([ally.id, angle, contrib])
            source_used.add(ally.id)
            shortfall -= contrib

    # ── Phase 2: Score all (source, target) pairs ─────────────────────────────
    candidates = []
    for mine in my_planets:
        if mine.id in source_used:
            continue
        base_res = _reserve(mine.ships, _turn) // 2 if behind_on_planets else _reserve(mine.ships, _turn)
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
            # For enemy planets: subtract confirmed third-party attackers from garrison
            if t.owner not in (-1, player):
                already += third_party_en.get(t.id, 0)
            net_garr = max(0.0, raw_garr - already)

            static_bonus  = 2.5 if _is_static(t) else 1.0
            neutral_bonus = 1.5 if t.owner == -1 else 1.0
            enemy_bonus   = (3.0 if few_neutrals else 2.0) if t.owner not in (-1, player) else 1.0
            prod_bonus    = 1.6 if (t.owner not in (-1, player) and losing_prod) else 1.0
            # 4-player: 1.4x bonus for targeting the weakest enemy player
            weak_bonus    = 1.4 if (t.owner == weakest_enemy) else 1.0
            # 4-player: 1.5x bonus when a third enemy is already attacking this planet
            dogpile_bonus = 1.5 if third_party_en.get(t.id, 0) > 0 else 1.0
            score = (static_bonus * neutral_bonus * enemy_bonus * prod_bonus
                     * weak_bonus * dogpile_bonus
                     * t.production / (dist * max(net_garr, 1.0)))
            candidates.append((score, mine.id, t.id, eta, tx, ty, net_garr))

    candidates.sort(key=lambda c: c[0], reverse=True)

    mine_avail = {p.id: max(0, p.ships - (_reserve(p.ships, _turn) // 2 if behind_on_planets else _reserve(p.ships, _turn)))
                  for p in my_planets if p.id not in source_used}

    # ── Phase 3: Greedy dispatch ───────────────────────────────────────────────
    for score, mine_id, t_id, eta, tx, ty, _ in candidates:
        if mine_id in source_used:
            continue
        avail = mine_avail.get(mine_id, 0)
        if avail < 1:
            continue

        t = pid_map[t_id]
        is_neutral = t.owner == -1

        already = friendly_en.get(t_id, 0) + sending_to.get(t_id, 0)
        if is_neutral:
            net_garr = max(0.0, t.ships - already)
        else:
            net_garr = max(0.0, t.ships + t.production * eta - already
                           - third_party_en.get(t_id, 0))

        if is_neutral:
            needed = int(net_garr) + 2
            if net_garr <= 0:
                continue
            if avail < needed:
                if avail > net_garr:
                    ships_to_send = avail
                else:
                    continue
            else:
                ships_to_send = needed
        else:
            needed = int(net_garr) + max(8, int(t.production * 6))
            if avail >= needed:
                ships_to_send = needed
            elif avail >= max(15, int(net_garr * 0.60)) and (losing_prod or few_neutrals):
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
        source_used.add(mine_id)
        sending_to[t_id] = sending_to.get(t_id, 0) + ships_to_send
        mine_avail[mine_id] -= ships_to_send

    return moves
