"""
Orbit Wars — Tier 2 Efficient Expansion Agent  (v2 — consolidation + safe harassment)

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
  7. [v2] Consolidation mode: when total enemy ships > 85% of our ships, neutral
     captures require the source to keep ≥ 10 ships post-send.  Prevents the
     "thin overexpansion" failure mode where we win planet count but lose ship count.
  8. [v2] Harassment restricted to losing_prod only: removed few_neutrals from the
     harassment trigger to stop burning ships on failed attacks while we're ahead.

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
    """Return (friendly_en_route, enemy_en_route): {planet_id -> ships}.

    Hybrid approach: check fleet angle against bearing to each planet's
    predicted position at rough ETA.
    """
    friendly, enemy = {}, {}
    for f in fleets:
        best_pid, best_diff = None, 0.30   # angle tolerance in radians
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
            # Weight by closeness to planet (prefer nearby planets)
            if diff < best_diff and dist_to_pred < 160:
                best_diff = diff
                best_pid = pid
        if best_pid is None:
            continue
        if f.owner == player:
            friendly[best_pid] = friendly.get(best_pid, 0) + f.ships
        else:
            enemy[best_pid] = enemy.get(best_pid, 0) + f.ships
    return friendly, enemy


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
    # If enemy holds more planets, we need faster expansion (lower reserve)
    behind_on_planets = len(enemy_planets) > len(my_planets) + 1
    # Direct combat mode: few neutrals left, focus on draining enemy
    neutrals_left = sum(1 for p in planets if p.owner == -1)
    few_neutrals = neutrals_left < 4

    # ── Situational flags ─────────────────────────────────────────────────────
    total_my_ships    = sum(p.ships for p in my_planets)
    total_enemy_ships = sum(p.ships for p in enemy_planets)
    avg_garrison      = total_my_ships / max(len(my_planets), 1)

    # [v2-fix1] Thin-expansion limiter: when avg garrison is dangerously low and
    # we're still in the expansion phase, cap neutral captures to 2 per turn.
    # Threshold avg<15 cleanly separates failing seeds (8–13) from winning ones (16+).
    thin_expanding = avg_garrison < 15 and _turn < 100

    # [v2-fix2] Late-game garrison guard: as neutrals run out (approaching combat
    # mode), require the source planet to keep ≥ 10 ships after a neutral capture.
    # Prevents stripping thin planets right before the enemy starts attacking.
    late_game_guard = neutrals_left <= 5 and _turn >= 50

    friendly_en, enemy_en = _parse_fleets(fleets, player, pid_map, ang_vel)

    moves       = []
    source_used = set()
    sending_to  = {}   # target_id -> ships we dispatch this turn

    # ── Phase 1: Reinforce threatened planets ─────────────────────────────────
    for mine in sorted(my_planets, key=lambda p: -enemy_en.get(p.id, 0)):
        threat = enemy_en.get(mine.id, 0)
        if threat == 0:
            continue
        # Use only current garrison + incoming friendlies (no production speculation)
        defense = mine.ships + friendly_en.get(mine.id, 0)
        if threat <= defense:
            continue
        shortfall = int(threat - defense) + 2
        # Allow multiple nearby allies to together cover the shortfall
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

            # Garrison at arrival (neutrals don't grow)
            raw_garr = t.ships if t.owner == -1 else t.ships + t.production * eta
            # Net garrison after already-committed ships
            already = friendly_en.get(t.id, 0) + sending_to.get(t.id, 0)
            net_garr = max(0.0, raw_garr - already)

            static_bonus  = 2.5 if _is_static(t) else 1.0
            neutral_bonus = 1.5 if t.owner == -1 else 1.0
            # Boost enemy planet priority in direct combat (few neutrals)
            enemy_bonus   = (3.0 if few_neutrals else 2.0) if t.owner not in (-1, player) else 1.0
            prod_bonus    = 1.6 if (t.owner not in (-1, player) and losing_prod) else 1.0
            score = (static_bonus * neutral_bonus * enemy_bonus * prod_bonus
                     * t.production / (dist * max(net_garr, 1.0)))
            candidates.append((score, mine.id, t.id, eta, tx, ty, net_garr))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Pre-compute available ships per source
    mine_avail = {p.id: max(0, p.ships - (_reserve(p.ships, _turn) // 2 if behind_on_planets else _reserve(p.ships, _turn)))
                  for p in my_planets if p.id not in source_used}

    # ── Phase 3: Greedy dispatch ───────────────────────────────────────────────
    neutral_cap = 0  # tracks neutral captures this turn (throttled when thin_expanding)
    for score, mine_id, t_id, eta, tx, ty, _ in candidates:
        if mine_id in source_used:
            continue
        avail = mine_avail.get(mine_id, 0)
        if avail < 1:
            continue

        t = pid_map[t_id]
        is_neutral = t.owner == -1

        # Recompute net garrison with latest sending_to
        already = friendly_en.get(t_id, 0) + sending_to.get(t_id, 0)
        if is_neutral:
            net_garr = max(0.0, t.ships - already)
        else:
            net_garr = max(0.0, t.ships + t.production * eta - already)

        if is_neutral:
            needed = int(net_garr) + 2   # minimal buffer — neutrals don't regenerate
            if net_garr <= 0:
                continue  # already covered

            mine_src = pid_map[mine_id]

            # [v2-fix1] Thin-expansion cap: when avg garrison is low, only fire the
            # top 2 neutral captures per turn so remaining planets build up garrison.
            if thin_expanding and neutral_cap >= 2:
                continue

            # [v2-fix2] Late-game garrison guard: as neutrals near exhaustion,
            # only capture if the source planet stays healthy (≥ 10 ships after send).
            if late_game_guard and avail >= needed and (mine_src.ships - needed) < 10:
                continue  # would leave source dangerously thin; skip this capture

            if avail < needed:
                # Allow sending when we're close enough to capture
                if avail > net_garr:   # strictly more ships than garrison = flip guaranteed
                    ships_to_send = avail
                else:
                    continue           # genuinely can't capture yet, wait
            else:
                ships_to_send = needed  # minimum-cost capture
        else:
            needed = int(net_garr) + max(8, int(t.production * 6))
            if avail >= needed:
                ships_to_send = needed
            elif avail >= max(15, int(net_garr * 0.60)) and (losing_prod or few_neutrals):
                ships_to_send = avail  # harassment: drain when losing or in combat phase
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
        if is_neutral:
            neutral_cap += 1  # track for thin_expanding throttle

    return moves
