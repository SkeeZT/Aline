"""
Exercise analyzers for AI Trainer application.

Available exercises:
- SquatExercise: Squat form analysis with ROM tracking
- PullupExercise: Pull-up form analysis with kipping detection
- PushupExercise: Push-up form analysis with body alignment tracking
- DipsExercise: Dips form analysis with shoulder shrug detection
- LungesExercise: Lunges form analysis with front/back leg tracking
- PlankExercise: Plank form analysis (isometric/time-based) with body alignment
- DeadliftExercise: Deadlift form analysis with back curvature monitoring
- OverheadPressExercise: Overhead press form analysis with lockout tracking
- BentOverRowExercise: Bent-over row form analysis with back curvature monitoring
- GluteBridgeExercise: Glute bridge form analysis with hip extension tracking
- WallSitExercise: Wall sit form analysis (isometric/time-based) with angle tracking

Dual Camera Support:
- DualCameraExerciseMixin: Mixin providing dual-camera analysis capabilities
  (knee valgus, hip alignment, arm symmetry, etc.)
"""

from engine.exercises.squat import SquatExercise
from engine.exercises.pullup import PullupExercise
from engine.exercises.pushup import PushupExercise
from engine.exercises.dips import DipsExercise
from engine.exercises.lunges import LungesExercise
from engine.exercises.plank import PlankExercise
from engine.exercises.deadlift import DeadliftExercise
from engine.exercises.overhead_press import OverheadPressExercise
from engine.exercises.bent_over_row import BentOverRowExercise
from engine.exercises.glute_bridge import GluteBridgeExercise
from engine.exercises.wall_sit import WallSitExercise
from engine.exercises.bench_press import BenchPressExercise
from engine.exercises.exercise_manager import ExerciseManager
from engine.exercises.dual_camera_mixin import DualCameraExerciseMixin, DualCameraMetrics

__all__ = [
    "SquatExercise",
    "PullupExercise", 
    "PushupExercise",
    "DipsExercise",
    "LungesExercise",
    "PlankExercise",
    "DeadliftExercise",
    "OverheadPressExercise",
    "BentOverRowExercise",
    "GluteBridgeExercise",
    "WallSitExercise",
    "BenchPressExercise",
    "ExerciseManager",
    "DualCameraExerciseMixin",
    "DualCameraMetrics",
]
