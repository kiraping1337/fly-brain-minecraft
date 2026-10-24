"""Тело мухи в мире: какая сущность и её геометрия."""

#тело
BODY_ENTITY = "minecraft:bee"
BODY_NBT = ""             # добавка к NBT при создании тела

# Тела для гонки
BODIES = {
    "bee":     ("minecraft:bee", "", 0.7, 0.3),
    "chicken": ("minecraft:chicken", "", 0.4, 0.644),
    "chick":   ("minecraft:chicken", "Age:-2147483647,", 0.2, 0.322),
    "allay":   ("minecraft:allay", "", 0.35, 0.36),
    "bat":     ("minecraft:bat", "BatFlags:0b,", 0.5, 0.45),
}


EYE_HEIGHT = 0.3
EYE_BOX_HALF = 0.8 * 0.7 / 2.0
BODY_HALF = 0.35


#источники данных

def apply_body(name):
    """Выбрать тело: сущность и геометрия удушья и краёв.

    Меняет модульные значения, которые функции читают в момент вызова, так
    что выбор при старте действует на весь мост.
    """
    global BODY_ENTITY, BODY_NBT, EYE_HEIGHT, EYE_BOX_HALF, BODY_HALF
    entity, extra, width, eye = BODIES[name]
    BODY_ENTITY, BODY_NBT = entity, extra
    EYE_HEIGHT = eye
    EYE_BOX_HALF = 0.8 * width / 2.0    # квадрат удушья - 0.8 от ширины
    BODY_HALF = width / 2.0
