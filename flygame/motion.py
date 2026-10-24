"""Движение тела: доворот рывками, ходьба, полёт, толчок от удара."""

import math

from .constants import (
    AVOID_TURN, BILATERAL_GAIN, BILATERAL_MIN, BILATERAL_PULL_DEG,
    CLIMB_RATE, CLIMB_SPEED, LANDING_NEAR_SPEED, LANDING_SPEED,
    MEAL_SEARCH_SPEED, ODOR_CAST, ODOR_CAST_FLOOR, ODOR_CAST_HOLD_S,
    ODOR_DIVE_RATE, ODOR_DIVE_RETREAT, PI_PULL_DEG, SACCADE_FLOOR,
    SACCADE_LEAK_S, SACCADE_MAX, SACCADE_MIN, SACCADE_RATE, SACCADE_URGENT,
    SEARCH_GROW_S, SEARCH_RUN_MAX_S, SEARCH_RUN_S, SETTLE_STEPS, SIDESTEP,
    SINK_RATE, STEP_FREQ_MIN, TAU_ACCEL, TAU_VERTICAL, TURN_RATE, WALK_NEAR,
    WALK_STALE_DEG)
from .geometry import bearing_deg, heading_vector, wrap_deg
from .state import Terrain
from .odor import odor_dive_allowed, odor_turn_bias
from .memory import (
    path_integral_pull, searching, tend_places, update_path_integral)


def push_body(body, threats, distance):
    """Толчок от удара"""
    if distance <= 0:
        return
    nearest = min(threats, key=lambda t: t.distance, default=None)
    if nearest is not None:
        angle = math.radians(body.yaw + nearest.azimuth_deg)
        body.x -= math.sin(angle) * distance
        body.z += math.cos(angle) * distance
    body.y += distance * 0.3


def apply_turn(body, motor, dt):
    """Доворот рывками. Общий и для полёта, и для ходьбы. Возвращает, на сколько градусов тело повернулось за этот такт."""
    needed = wrap_deg(motor.escape_deg)
    if motor.retreat <= ODOR_DIVE_RETREAT and abs(body.odor_side) > BILATERAL_MIN:
        turn = min(BILATERAL_PULL_DEG, BILATERAL_GAIN * abs(body.odor_side))
        needed = wrap_deg(needed - math.copysign(turn, body.odor_side))
    if motor.retreat <= ODOR_DIVE_RETREAT:
        pull = path_integral_pull(body)
        if pull:
            needed = wrap_deg(needed + max(-PI_PULL_DEG,
                                           min(PI_PULL_DEG, pull)))
    body.cast_hold = max(0.0, body.cast_hold - dt)
    body.meal_search = max(0.0, body.meal_search - dt)
    tend_places(body, dt)

    body.search_hold = max(0.0, body.search_hold - dt)
    if searching(body):
        body.search_time += dt
    else:
        body.search_time = 0.0
        body.search_hold = 0.0

    if body.saccade == 0.0:
        body.turn_debt = (body.turn_debt * math.exp(-dt / SACCADE_LEAK_S)
                          + needed * dt)
        bias = odor_turn_bias(body)
        threshold = SACCADE_MIN * bias
        urgent = abs(needed) >= SACCADE_URGENT
        if urgent or abs(body.turn_debt) >= threshold:
            floor = (ODOR_CAST_FLOOR if bias == ODOR_CAST and not urgent
                     else SACCADE_FLOOR)
            amplitude = needed if urgent else math.copysign(
                max(abs(body.turn_debt), floor), body.turn_debt)
            body.saccade = max(-SACCADE_MAX, min(SACCADE_MAX, amplitude))
            body.turn_debt = 0.0
            body.saccades += 1
            if bias == ODOR_CAST and not urgent:
                body.cast_hold = ODOR_CAST_HOLD_S
            if not urgent and searching(body):
                body.search_hold = min(SEARCH_RUN_MAX_S, SEARCH_RUN_S
                                       * (1.0 + body.search_time / SEARCH_GROW_S))

    step = 0.0
    if body.saccade != 0.0:
        step = math.copysign(min(abs(body.saccade), SACCADE_RATE * dt),
                             body.saccade)
        body.saccade -= step
        if abs(body.saccade) < 0.5:
            body.saccade = 0.0
    body.yaw = (body.yaw + step) % 360.0
    return needed, step


def walk(body, motor, dt, terrain, args, settling=False, hold=False):
    """Ходьба: тело продвигается дискретными шагами, а не скользит"""

    turns_before = body.saccades
    apply_turn(body, motor, dt)
    if settling and body.saccades > turns_before:
        body.pending_steps = SETTLE_STEPS

    body.vy = 0.0
    if terrain.clear_below:
        drop = min(LANDING_SPEED * 0.5 * dt, 0.06)
        if terrain.at is not None:
            known_floor = math.floor(terrain.at[1] - 1.0)
            drop = min(drop, max(0.0, body.y - known_floor))
        body.y -= drop

    if hold:
        body.pending_steps = 0
        body.vx = body.vz = 0.0
        return

    top = max(args.step_rate, STEP_FREQ_MIN)
    frequency = STEP_FREQ_MIN + motor.thrust * (top - STEP_FREQ_MIN)
    if body.meal_search > 0.0:
        frequency *= MEAL_SEARCH_SPEED      # поиск вокруг еды, а не орбита

    forward_x, forward_z = heading_vector(body.yaw)
    body.travel_yaw = body.yaw
    if settling:
        if body.pending_steps <= 0:
            body.vx = body.vz = 0.0
            return
        frequency *= 0.8

    if (terrain.probe_yaw is not None
            and abs(wrap_deg(body.yaw - terrain.probe_yaw)) > WALK_STALE_DEG):
        body.want_recheck = True
        body.vx = body.vz = 0.0
        return

    average = args.step_length * frequency
    body.vx, body.vz = forward_x * average, forward_z * average

    room = terrain.walk_room if terrain.at is not None else None
    if room is not None:
        probe_x, probe_z = heading_vector(terrain.probe_yaw or 0.0)

    body.step_phase += frequency * dt
    while body.step_phase >= 1.0:
        body.step_phase -= 1.0
        body.steps += 1
        if terrain.wet_ahead:
            body.yaw = (body.yaw + math.copysign(60.0, motor.lateral or 1.0)) % 360.0
            break

        blocked_ahead = (not terrain.free_ahead
                         or (terrain.edge_left and terrain.edge_right))
        if not blocked_ahead and terrain.edge_left != terrain.edge_right:
            left_x, left_z = forward_z, -forward_x
            away = -1.0 if terrain.edge_left else 1.0
            body.x += left_x * SIDESTEP * away
            body.z += left_z * SIDESTEP * away
        if blocked_ahead:
            if terrain.step_up and not terrain.free_ahead:
                if terrain is not body.step_terrain:
                    if room is not None:
                        target_x = terrain.at[0] + probe_x * WALK_NEAR
                        target_z = terrain.at[2] + probe_z * WALK_NEAR
                        move_x, move_z = target_x - body.x, target_z - body.z
                        body.y = terrain.at[1] + 1.0
                    else:
                        move_x, move_z = forward_x * WALK_NEAR, forward_z * WALK_NEAR
                        body.y += 1.0
                    body.x += move_x
                    body.z += move_z
                    update_path_integral(body, move_x, move_z, dt)
                    body.step_terrain = terrain
                    body.want_recheck = True
                    break
            else:
                body.yaw = (body.yaw + math.copysign(60.0, motor.lateral or 1.0)) % 360.0
                break
        if room is not None:
            along = ((body.x - terrain.at[0]) * probe_x
                     + (body.z - terrain.at[2]) * probe_z)

            if (along + args.step_length > room
                    or math.floor(body.y) < math.floor(terrain.at[1])):

                body.steps -= 1
                body.step_phase = 1.0
                body.want_recheck = True
                body.vx = body.vz = 0.0
                break
        body.x += forward_x * args.step_length
        body.z += forward_z * args.step_length
        update_path_integral(body, forward_x * args.step_length,
                             forward_z * args.step_length, dt)
        if settling:
            body.pending_steps -= 1
            if body.pending_steps <= 0:
                break


def steer(body, motor, dt, home, args, terrain=None, state="flying",
          hold=False):
    """Поворот и тяга -> новое положение тела."""
    needed, step = apply_turn(body, motor, dt)

    desired = body.yaw + wrap_deg(needed - step)
    body.travel_yaw += wrap_deg(desired - body.travel_yaw) * min(
        1.0, dt / max(args.turn_smoothing, 1e-3))
    body.travel_yaw %= 360.0
    travel_x, travel_z = heading_vector(body.travel_yaw)

    blend = min(1.0, dt / TAU_ACCEL)
    wanted = args.speed * motor.thrust
    if body.meal_search > 0.0:
        wanted *= MEAL_SEARCH_SPEED         # то же и в полёте
    body.cruise += (wanted - body.cruise) * blend
    target_vx, target_vz = travel_x * body.cruise, travel_z * body.cruise

    body.vx, body.vz = target_vx, target_vz

    #наши добавки, а не мозг
    terrain = terrain or Terrain()

    if terrain.stuck:
        body.vy = CLIMB_RATE * 2.0
        body.y += body.vy * dt
        body.vx *= 0.7
        body.vz *= 0.7
        body.x += body.vx * dt
        body.z += body.vz * dt
        update_path_integral(body, body.vx * dt, body.vz * dt, dt)
        return

    if state == "perched":
        walk(body, motor, dt, terrain, args, settling=True, hold=hold)
        return

    if state == "walking":
        walk(body, motor, dt, terrain, args, hold=hold)
        return

    if state == "landing":
        body.vx *= 0.85
        body.vz *= 0.85

        speed = LANDING_SPEED if terrain.above_ground else LANDING_NEAR_SPEED
        body.vy = -speed
        body.y -= speed * dt
        body.x += body.vx * dt
        body.z += body.vz * dt
        update_path_integral(body, body.vx * dt, body.vz * dt, dt)
        return

    if not terrain.free_ahead:
        body.yaw += math.copysign(AVOID_TURN, motor.lateral or 1.0) * TURN_RATE * dt
        forward_x, forward_z = heading_vector(body.travel_yaw)
        into_wall = body.vx * forward_x + body.vz * forward_z
        if into_wall > 0.0:
            body.vx -= into_wall * forward_x
            body.vz -= into_wall * forward_z
        body.vx *= 0.85
        body.vz *= 0.85


    target_vy = motor.climb * CLIMB_SPEED - SINK_RATE
    if odor_dive_allowed(body, motor) and terrain.above_ground:
        target_vy -= ODOR_DIVE_RATE
    if not terrain.above_ground:
        target_vy = max(target_vy, CLIMB_RATE)   # земля близко, не снижаемся
    elif body.y < home[1] - args.floor:
        target_vy = max(target_vy, 0.0)          # ниже этого не проваливаемся
    body.vy += (target_vy - body.vy) * min(1.0, dt / TAU_VERTICAL)
    body.y += body.vy * dt

    if args.leash > 0:
        offset = math.dist((body.x, 0, body.z), (home[0], 0, home[2]))
        if offset > args.leash:
            toward_home = bearing_deg(home[0] - body.x, home[2] - body.z)
            pull = min(1.0, (offset - args.leash) / 8.0)
            body.yaw += wrap_deg(toward_home - body.yaw) * pull * 0.25
            body.vx *= 1.0 - 0.5 * pull
            body.vz *= 1.0 - 0.5 * pull

    body.x += body.vx * dt
    body.z += body.vz * dt
    update_path_integral(body, body.vx * dt, body.vz * dt, dt)
