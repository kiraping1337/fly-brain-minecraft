"""Калибровка порогов поведения по покою мозга"""

from dataclasses import dataclass

import numpy as np

from flybrain import (Circuit, FlyBrain, LoomingEncoder, MotorDecoder,
                      OdorEncoder, VisualField, WindEncoder)

from .constants import (
    CONTROL_HZ, HUNGRY_RESTLESS, TAU_AROUSAL_FALL, TAU_AROUSAL_RISE)


def measure_idle_arousal(brain, decoder, circuit, seconds=4.0):
    """Каким получается ВОЗБУЖДЕНИЕ в покое, а не мгновенная тяга"""
    silence = np.zeros(circuit.input_idx.size)
    dt = 1.0 / CONTROL_HZ
    arousal, samples, sideways = 0.0, [], []
    decoder.lateral_baseline = 0.0
    brain.reset()
    for step in range(int(seconds * CONTROL_HZ)):
        brain.advance(silence, duration_ms=dt * 1000.0)
        command = decoder.decode(brain.output_rates_hz)
        tau = TAU_AROUSAL_RISE if command.thrust > arousal else TAU_AROUSAL_FALL
        arousal += (command.thrust - arousal) * min(1.0, dt / tau)
        if step > seconds * CONTROL_HZ / 2:
            samples.append(arousal)
            sideways.append(command.lateral)
    brain.reset()
    decoder.lateral_baseline = float(np.mean(sideways)) if sideways else 0.0
    return float(np.mean(samples)) if samples else 0.0


def tune_hungry_noise(brain, decoder, circuit, perch_thrust,
                      target=HUNGRY_RESTLESS, seconds=8.0, ceiling_hz=4.0,
                      rounds=6):
    """Сколько шума прибавить голодной мухе"""
    base = brain.spontaneous_hz
    silence = np.zeros(circuit.input_idx.size)
    dt = 1.0 / CONTROL_HZ
    warmup = int(1.5 * CONTROL_HZ)
    low, high = 0.0, float(ceiling_hz)
    for _ in range(rounds):
        middle = 0.5 * (low + high)
        brain.spontaneous_hz = base + middle
        brain.reset()
        arousal, above, total = 0.0, 0, 0
        for step in range(int(seconds * CONTROL_HZ)):
            brain.advance(silence, duration_ms=dt * 1000.0)
            command = decoder.decode(brain.output_rates_hz)
            tau = TAU_AROUSAL_RISE if command.thrust > arousal else TAU_AROUSAL_FALL
            arousal += (command.thrust - arousal) * min(1.0, dt / tau)
            if step >= warmup:
                total += 1
                above += arousal >= perch_thrust
        if above / max(total, 1) > target:
            high = middle
        else:
            low = middle
    brain.spontaneous_hz = base
    brain.reset()
    return 0.5 * (low + high)


@dataclass
class Calibration:
    """Мозг с органами чувств и пороги, замеренные на нём при старте."""
    circuit: object
    visual_field: object
    encoder: object
    wind: object
    odor: object
    brain: object
    decoder: object
    odor_rest: float        # уровень доли на чистом воздухе (x1.5)
    idle_arousal: float     # возбуждение покоя
    perch_thrust: float     # садится ниже этого
    takeoff_thrust: float   # взлетает выше этого
    base_noise: float       # фоновый шум сытой, Гц
    hungry_extra: float     # прибавка шума у голодной, Гц
    perch_hungry: float     # порог посадки голодной из полёта


def calibrate(args):
    """Собрать мозг и откалибровать пороги: всё, что main делал до цикла."""
    circuit = Circuit.load(args.circuit)
    visual_field = VisualField(circuit)
    encoder = LoomingEncoder(visual_field, k_saturation=args.k_sat)
    wind = WindEncoder(circuit, k_saturation=args.k_wind)
    odor = OdorEncoder(circuit)
    auto_noise = str(args.spontaneous).lower() == "auto"
    brain = FlyBrain(circuit,
                     track_all=bool(args.record),
                     watch_idx=odor.level_idx if odor.active else None,
                     calibration_visibility=visual_field.visibility(0.0),
                     spontaneous_hz=0.0 if auto_noise else float(args.spontaneous))
    decoder = MotorDecoder(circuit)

    print(circuit.describe())
    print(f"Калибровка: GAIN = {brain.gain:.4f}, чувствительность {args.k_sat}")
    print("Механосенсорика: " + (f"{int(circuit.mechano_mask.sum())} нейронов "
                                 f"джонстонова органа"
                                 if wind.active else "нет в схеме"))
    if auto_noise:
        print("Подбираю фоновый шум...", end="", flush=True)
        print(f"\r Фоновый шум: {brain.autotune_spontaneous(args.idle_hz):.1f} Гц"
              f" (подобран под размер схемы)")
    else:
        print(f"Фоновый шум: {brain.spontaneous_hz:.1f} Гц")

    brain.reset()
    brain.advance(np.zeros(circuit.input_idx.size), duration_ms=800.0)
    print(f"Фон на выходе: {decoder.calibrate_baseline(brain.output_rates_hz):.1f} Гц")

    odor_rest = (odor.odor_level(brain.watch_rates_hz) * 1.5
                 if odor.active else 0.0)
    idle_arousal = measure_idle_arousal(brain, decoder, circuit)
    perch_thrust = idle_arousal * args.perch_factor
    takeoff_thrust = idle_arousal * args.takeoff_factor
    print(f"Возбуждение в покое {idle_arousal:.2f}: садится ниже "
          f"{perch_thrust:.2f}, выше - уходит пешком, "
          f"взлетает при подъёме {args.takeoff_climb:.2f} "
          f"или возбуждении {takeoff_thrust:.2f}")

    base_noise = brain.spontaneous_hz
    hunger_on = args.hunger > 0 or args.hunger_minutes > 0
    if str(args.hungry_noise).lower() != "auto":
        hungry_extra = float(args.hungry_noise)
    elif hunger_on:
        print("Подбираю шум голодной мухи...", end="", flush=True)
        hungry_extra = tune_hungry_noise(brain, decoder, circuit, perch_thrust)
    else:
        hungry_extra = 0.0

    perch_hungry = perch_thrust
    if hungry_extra > 0.0:
        base_noise_saved = brain.spontaneous_hz
        brain.spontaneous_hz = base_noise_saved + hungry_extra
        perch_hungry = max(perch_thrust, measure_idle_arousal(brain, decoder, circuit)
                           * args.perch_factor)
        brain.spontaneous_hz = base_noise_saved
        brain.reset()
    print(f"\rГолод: у голодной шум +{hungry_extra:.2f} Гц, вход угрозы "
          f"×{1.0 - args.boldness:.2f}; на старте {args.hunger:.2f}, "
          + (f"голодает за {args.hunger_minutes:.0f} мин сидя"
             if args.hunger_minutes > 0 else "не растёт"))
    brain.reset()
    return Calibration(circuit=circuit, visual_field=visual_field, encoder=encoder, wind=wind,
                       odor=odor, brain=brain, decoder=decoder, odor_rest=odor_rest,
                       idle_arousal=idle_arousal, perch_thrust=perch_thrust, takeoff_thrust=takeoff_thrust,
                       base_noise=base_noise, hungry_extra=hungry_extra, perch_hungry=perch_hungry)
