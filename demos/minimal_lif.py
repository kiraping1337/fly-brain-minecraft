"""
Мини-версия схемы убегания.

Что здесь есть:
  - 20 "LC-нейронов" (детекторы приближения) - они генерируют спайки
    с частотой, зависящей от того, насколько быстро объект растёт в поле зрения.
  - 2 "DNp01" (гигантское волокно, команда "прыгай") - это настоящие
    LIF-нейроны, которые копят заряд от LC и спайкают при переполнении.
  - Стимул: объект летит на муху. Считаем угловой размер и скорость расширения.

Запуск:  python demos/minimal_lif.py
Результат: окно с графиками + output/minimal_lif.png

"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "output" / "minimal_lif.png"

#параметры


#время
DT = 0.1            # шаг симуляции, мс
T_TOTAL = 1800.0    # сколько всего симулируем, мс
n_steps = int(T_TOTAL / DT)
time = np.arange(n_steps) * DT

#нейроны (ориентиры из статьи Shiu et al. 2024) ---
V_REST = -52.0      # потенциал покоя, мВ
V_THRESH = -45.0    # порог спайка, мВ
V_RESET = -52.0     # куда падает после спайка, мВ
TAU_MEM = 20.0      # постоянная времени мембраны, мс
T_REFRAC = 2.2      # рефрактерный период, мс
W_SYN = 0.275       # вклад ОДНОГО синапса в мВ

#схема
N_LC = 20           # сколько LC-нейронов
N_DN = 2            # DNp01 левый и правый
SYN_LC_DN = 2       # синапсов от каждого LC к каждому DN

#стимул: объект летит на нас
OBJ_SIZE = 30.0     # размер объекта, мм
DIST_START = 300.0  # начальная дистанция, мм
SPEED = 200.0       # скорость сближения, мм/с
T_ONSET = 200.0     # когда объект начинает двигаться, мс

#энкодер: как скорость расширения превращается в частоту спайков
F_MAX = 150.0       # максимальная частота LC, Гц
K_SAT = 1.0         # при какой скорости расширения (рад/с) наступает потолок

rng = np.random.default_rng(42)   # фиксируем сид, чтобы результат повторялся


#стимул
# Угловой размер: theta = 2 * arctan(размер / (2 * дистанция))
# Нейроны LC реагируют НЕ на дистанцию, а на d(theta)/dt - скорость роста.

t_sec = time / 1000.0
t_move = np.clip(t_sec - T_ONSET / 1000.0, 0, None)      # время с начала движения
distance = np.maximum(DIST_START - SPEED * t_move, 5.0)  # мм, не даём уйти в 0

theta = 2.0 * np.arctan(OBJ_SIZE / (2.0 * distance))     # угловой размер, рад
dtheta_dt = np.gradient(theta, t_sec)                    # скорость расширения, рад/с

# Энкодер: скорость расширения -> частота спайков LC
lc_rate = F_MAX * np.clip(dtheta_dt / K_SAT, 0.0, 1.0)   # Гц


#матрица связей

W = np.zeros((N_LC, N_DN))
W[:, :] = SYN_LC_DN
sign = np.ones(N_LC)          # все LC-нейроны холинергические => возбуждающие

psp = (W * sign[:, None]) * W_SYN


# симуляция

V = np.full(N_DN, V_REST)          # текущий потенциал DN-нейронов
refrac_until = np.full(N_DN, -1.0) # до какого времени нейрон "глухой"

lc_spike_times, lc_spike_ids = [], []
dn_spike_times, dn_spike_ids = [], []
V_trace = np.zeros((n_steps, N_DN))

for step in range(n_steps):
    t = time[step]

    p = lc_rate[step] * DT / 1000.0
    lc_spikes = rng.random(N_LC) < p

    if lc_spikes.any():
        idx = np.flatnonzero(lc_spikes)
        lc_spike_times.extend([t] * len(idx))
        lc_spike_ids.extend(idx)

    V += DT * (V_REST - V) / TAU_MEM
    V += lc_spikes @ psp

    still_refrac = t < refrac_until
    V[still_refrac] = V_RESET

    fired = (V >= V_THRESH) & ~still_refrac
    if fired.any():
        idx = np.flatnonzero(fired)
        dn_spike_times.extend([t] * len(idx))
        dn_spike_ids.extend(idx)
        V[fired] = V_RESET
        refrac_until[fired] = t + T_REFRAC

    V_trace[step] = V

lc_spike_times = np.array(lc_spike_times)
lc_spike_ids = np.array(lc_spike_ids)
dn_spike_times = np.array(dn_spike_times)


# результат
t_collision = T_ONSET + (DIST_START - OBJ_SIZE) / SPEED * 1000.0

print("=" * 55)
print(f"Спайков LC:  {len(lc_spike_times)}")
print(f"Спайков DN:  {len(dn_spike_times)}")

if len(dn_spike_times):
    t_first = dn_spike_times.min()
    print(f"Первый спайк DN:      {t_first:.0f} мс")
    print(f"Латентность от старта: {t_first - T_ONSET:.0f} мс")
    print(f"Запас до столкновения: {t_collision - t_first:.0f} мс  <- муха успела")
else:
    print("DN молчит. Увеличьте W_SYN, SYN_LC_DN или F_MAX.")
print("=" * 55)

# графики

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

ax[0].plot(time, np.degrees(theta), color="#1f77b4", label="угловой размер, °")
ax[0].set_ylabel("размер объекта, °")
ax0b = ax[0].twinx()
ax0b.plot(time, lc_rate, color="#ff7f0e", ls="--", label="частота LC, Гц")
ax0b.set_ylabel("частота LC, Гц", color="#ff7f0e")
ax[0].set_title("Стимул: объект приближается")

if len(lc_spike_times):
    ax[1].scatter(lc_spike_times, lc_spike_ids, s=1.5, color="#2ca02c")
ax[1].set_ylabel("LC-нейрон №")
ax[1].set_title("Спайки LC4 / LPLC2 (детекторы приближения)")

ax[2].plot(time, V_trace[:, 0], color="#d62728", lw=0.8)
ax[2].axhline(V_THRESH, color="k", ls=":", lw=1, label="порог")
for ts in dn_spike_times:
    ax[2].axvline(ts, color="#d62728", alpha=0.25, lw=1)
ax[2].set_ylabel("потенциал DNp01, мВ")
ax[2].set_xlabel("время, мс")
ax[2].set_title("DNp01 (гигантское волокно): красные линии = команда ПРЫГАЙ")
ax[2].legend(loc="upper left")

for a in ax:
    a.axvline(T_ONSET, color="gray", ls="--", lw=1)
    a.axvline(t_collision, color="black", ls="-", lw=1.5, alpha=0.6)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130)
print(f"График сохранён: {OUT}")
plt.show()
