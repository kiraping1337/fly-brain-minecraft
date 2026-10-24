"""MinecraftWorld - мир через RCON, собранный из пяти частей."""

from flygame.constants import FOOD_ITEMS, WALK_NEAR, WALK_FAR
from flygame.state import Terrain

from . import gateway, ident, shape
from .constants import (
    PASSABLE_BLOCKS, POLL_INTERVAL_S, RCON_HOST, RCON_PASSWORD, RCON_PORT,
    SOLID_BLOCKS)
from .link import FairLock, LinkPart
from .rcon import open_rcon
from .body import BodyPart
from .creatures import CreaturesPart
from .terrain import TerrainPart
from .food import FoodPart


class MinecraftWorld(LinkPart, BodyPart, CreaturesPart, TerrainPart, FoodPart):
    def __init__(self, host=RCON_HOST, port=RCON_PORT, password=RCON_PASSWORD):
        if gateway.GATEWAY is not None:
            self._connection = gateway.GatewayConnection(gateway.GATEWAY, ident.NAME)
        else:
            self._connection = open_rcon(host, password, port)
        self._connection.connect()
        self._talking = FairLock()
        self._effects_path = None
        self._invisible = False
        self.use_motion = False
        self.spawn_at = None        # (x, y, z, yaw) для spawn_body; None - перед игроком
        self._poll = 0
        self._last_terrain = Terrain()

        self._ahead_ray = [(0.0, 1.5)]
        self._walking = False
        self._moving = False

        self.poll_period = POLL_INTERVAL_S
        self._near = (0.0, 0.0, WALK_NEAR, shape.BODY_HALF, 0.0,
                      tuple((0.0, d) for d in WALK_FAR))
        self._passable = PASSABLE_BLOCKS
        self._solid = SOLID_BLOCKS
        self._pass_cache = {}
        self._pass_chunks = None
        self._scoreboard_ready = False
        self._last_place = None

        self.body_at = None

        self.multi_type_ok = True
        self._shared_link = None
        self.see_flies = True

        self.food_items = tuple(FOOD_ITEMS)
        self.food_only = False
        self._shared_seq = 0
        self._shared_last = None

        self.shared_stats = dict(taken=0, missed=0, whole=0, creatures=0,
                                 projectiles=0, odor=0, odor_calls=0)
        self.last_answer = ""
        self.mismatched = 0
        self.bad_answers = []
