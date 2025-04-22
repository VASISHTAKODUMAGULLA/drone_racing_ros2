import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Tkagg')


class Mixin:

    def plot(self):
        # self.get_logger().info('Plotting')
        plt.clf()
        plt.ion()
        if self.sim:
            plt.plot([x[0] for x in self.trajectory_odom], [x[1] for x in self.trajectory_odom], label='ODOM')
        else:
            plt.plot([x[0] for x in self.trajectory_odom], [x[1] for x in self.trajectory_odom], label='ODOM')
        plt.title('Trajectory')
        plt.figure(1, figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.pause(0.001)
