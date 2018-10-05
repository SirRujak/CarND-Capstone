#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
f
Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
   def __init__(self):
       rospy.init_node('waypoint_updater')

       rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
       rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
       rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
       rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_callback)

       # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


       self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

       # TODO: Add other member variables you need below

       self.pose = None
       self.current_linear_velocity = None
       self.base_waypoints = None
       self.waypoints_2d = None
       self.waypoint_tree = None
       self.stopline_wp_idx = None
       self.currently_slowing = False
       self.slow_start = 0
       self.slow_end = 0
       self.start_dist = 0.0
       self.stop_dist = 10.0

       self.loop()


   def loop(self):
       rate = rospy.Rate(50)
       while not rospy.is_shutdown():
           if self.pose and self.base_waypoints and self.waypoint_tree:
               # Get closest waypoint.
               ##closest_waypoint_idx = self.get_closest_waypoint_idx()
               ##self.publish_waypoints(closest_waypoint_idx)
               self.publish_waypoints()
               #rospy.logwarn("test2")
           rate.sleep()

   def velocity_callback(self, msg):
        """
        /current_velocity topic callback handler.
        msg : geometry_msgs.msg.TwistStamped

        Updates state:
        - current_linear_velocity
        - current_angular_velocity
        """
        self.current_linear_velocity = msg.twist.linear.x
        #self.current_angular_velocity = msg.twist.angular.z

   def get_closest_waypoint_idx(self):
       x = self.pose.pose.position.x
       y = self.pose.pose.position.y
       closest_idx = self.waypoint_tree.query([x, y], 1)[1]

       # Check if closest waypoint is ahead or behind vehicle.
       closest_coord = self.waypoints_2d[closest_idx]
       prev_coord = self.waypoints_2d[closest_idx - 1]

       # Equation for hyperplane through closest_coords.
       cl_vect = np.array(closest_coord)
       prev_vect = np.array(prev_coord)
       pos_vect = np.array([x, y])

       val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

       if val > 0:
           closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
       return closest_idx

   def publish_waypoints(self):
       '''
       lane = Lane()
       lane.header = self.base_waypoints.header
       lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
       self.final_waypoints_pub.publish(lane)
       '''
       final_lane = self.generate_lane()
       self.final_waypoints_pub.publish(final_lane)

   def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx:
            #rospy.logwarn(self.stopline_wp_idx)
            if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
                #rospy.logwarn("Test")
                lane.waypoints = base_waypoints
                self.currently_slowing = False
            else:
                #rospy.logwarn("Need to stop.")
                lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
                self.currently_slowing = True
        else:
            lane.waypoints = base_waypoints
        return lane

   def h00(self, t):
       return 2*t*t*t - 3*t*t + 1

   def t(self, start_position, end_position, current_position):
       return (current_position - start_position)/(end_position - start_position +0.001)
    
   def target_velocity(self, start_position, end_position, current_position,
                             start_velocity):
       return self.h00(self.t(start_position, end_position, current_position)) * start_velocity

   def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        current_vel = waypoints[0].twist.twist.linear.x

        if not self.currently_slowing:
            rospy.logwarn("Just calculated stopping curve!")
            self.slow_start = self.pose.pose.position.x

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            self.slow_start = self.distance(waypoints, 0, 0)
            self.slow_end = self.distance(waypoints, 0, stop_idx)
            #self.slow_end = waypoints[stop_idx].pose.pose.position.x
            self.currently_slowing = True
            self.current_vel = self.current_linear_velocity
            if self.current_vel < 2.0:
                self.current_vel = 3.0

            rospy.logwarn("Range to stop is: " + str(self.slow_end))
            rospy.logwarn("Current velocity is: " + str(self.current_vel))
        ##total_distance = self.distance(waypoints, 0, stop_idx)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            current_position = p.pose.pose.position.x
            stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
            #dist = self.distance(waypoints, i, stop_idx)
            #vel = math.sqrt(2 * MAX_DECEL * dist)
            dist = self.slow_end - self.distance(waypoints, i , stop_idx)
            vel = self.target_velocity(self.slow_start, self.slow_end, dist, self.current_vel)
            if vel < 1.:
                vel = 0.
            #rospy.logwarn(vel)

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        #rospy.logwarn(temp[0].twist.twist.linear.x)
        #rospy.logwarn(self.slow_end)
        return temp
        '''  
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            ## Two waypoints back from line so front of car stops at line.
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            ## FIXME: Needs a differentiable function here.
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp
        '''

   def pose_cb(self, msg):
       # TODO: Implement
       self.pose = msg

   def waypoints_cb(self, waypoints):
       # TODO: Implement
       self.base_waypoints = waypoints
       if not self.waypoints_2d:
           self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
           self.waypoint_tree = KDTree(self.waypoints_2d)

   def traffic_cb(self, msg):
       # TODO: Callback for /traffic_waypoint message. Implement
       self.stopline_wp_idx = msg.data

   def obstacle_cb(self, msg):
       # TODO: Callback for /obstacle_waypoint message. We will implement it later
       pass

   def get_waypoint_velocity(self, waypoint):
       return waypoint.twist.twist.linear.x

   def set_waypoint_velocity(self, waypoints, waypoint, velocity):
       waypoints[waypoint].twist.twist.linear.x = velocity

   def distance(self, waypoints, wp1, wp2):
       dist = 0
       dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
       for i in range(wp1, wp2+1):
           dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
           wp1 = i
       return dist


if __name__ == '__main__':
   try:
       WaypointUpdater()
   except rospy.ROSInterruptException:
       rospy.logerr('Could not start waypoint updater node.')
