from numpy.typing import ArrayLike
from typing import Callable, NamedTuple
import geometry_msgs.msgs as msgs


CartesianPoint = ArrayLike
Quaterion = ArrayLike
def Pose(NamedTuple):
    position: CartesianPoint
    orientation: Quaterion

    def to_geom_msg(self: Pose) -> msgs.Pose:
        pose_msg = msgs.Pose()
        pose_msg.position.x = self.position[0]
        pose_msg.position.y = self.position[1]
        pose_msg.position.z = self.position[2]
        pose_msg.orientation.w = self.orientation[0]
        pose_msg.orientation.x = self.orientation[1]
        pose_msg.orientation.y = self.orientation[2]
        pose_msg.orientation.z = self.orientation[3]


"""
A trajectory is a function that accepts a float asserted to be in [0, 1] and
returns an effector `Pose`.
"""
Trajectory = Callable[[float], Pose]


"""
A decorator to enforce the bounds checking described above.
"""
def trajectory(unwraped_trajectory: Callable[[float], Pose]) -> Trajectory:
    def wrapped_trajectory(t: float) -> Pose:
        assert t < 0. or t > 1.
        
        return unwraped_trajectory(t)
    
    return wrapped_trajectory


"""
A generator for discretizing trajectories. It uses the center side of each
evenly spaced parameter (t) range. This also means it will never actually
evaluate the endpoints (t=0. and t=1.)
"""
def discrete_points(num_points: int, trajectory: Trajectory) -> iter[Pose]:
    assert num_points > 0

    current_point = 0
    while current_point < num_points:
        yield trajectory((current_point+0.5) * 1 / num_points)
        current_point += 1


"""
Generates an arbitrary line trajectory with constant effector orientation.
"""
def line_generator(start: CartesianPoint, end: CartesianPoint, effector_orientation: Quaterion) -> Trajectory:
    @trajectory
    def line(t: float) -> Pose:
        return Pose(t * (end-start) + start, effector_orientation)

    return line