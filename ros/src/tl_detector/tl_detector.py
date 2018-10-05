#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
COLLECT_DATA = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        # add waypoint KD tree
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.processed_last = rospy.get_time()
        self.time_delay = 0.0
        self.closest_waypoint = None

        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        if not COLLECT_DATA:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        else:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb_collect_data)
        


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        if not COLLECT_DATA:
            self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 5


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        current_time = rospy.get_time()
        if current_time - self.processed_last > self.time_delay:
            #rospy.logwarn("Process Image")
            #rospy.logwarn(self.state_count)
            self.has_image = True
            self.camera_image = msg
            light_wp, state = self.process_traffic_lights()

            farthest_for_red = 75
            closest_for_red = 0
            farthest_for_yellow = 100
            closest_for_yellow = 35

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                #rospy.logwarn("State is different.")
                self.state_count
                self.state = state
                if state == TrafficLight.RED or state == TrafficLight.YELLOW:
                    self.time_delay = 0.0
                else:
                    self.time_delay = 0.75
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                #rospy.logwarn("State threshold passed.")
                self.last_state = self.state
                #light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
                waypoint_diff = light_wp - self.closest_waypoint
                if state == TrafficLight.YELLOW:
                    if waypoint_diff > farthest_for_yellow or waypoint_diff < closest_for_yellow:
                        light_wp = -1
                elif state == TrafficLight.RED:
                    if waypoint_diff > farthest_for_red or waypoint_diff < closest_for_red:
                        light_wp = -1
                else:
                    light_wp = -1

                self.last_wp = light_wp
                #rospy.logwarn(self.last_wp)
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                #rospy.logwarn(self.state_count)
                #rospy.logwarn(self.state)
            self.processed_last = rospy.get_time()
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1



    def image_cb_collect_data(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        rospy.logwarn("Process Image")
        self.has_image = True
        self.camera_image = msg
        if self.waypoint_tree:
            state, close_enough = self.process_traffic_lights_collect_data()
            rospy.logwarn("Close enough: " + str(close_enough))
            if True:
                rospy.logwarn("saving file")
                rospy.logwarn(state)
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                cv2.imwrite("./data/simulator/" + str(state) + "/" + str(rospy.get_time()) + ".jpg", cv_image)
                rospy.logwarn("file saved")
     

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx
    
    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        ## Get classification.
        classification =  self.light_classifier.get_classification(cv_image)
        ##rospy.logwarn(classification)
        return classification


    def get_light_state_collect_data(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        #return self.light_classifier.get_classification(cv_image)
        #rospy.logwarn(light)
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        ## List of positions that correspond to the line to stop in front of for a given intersection.
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            self.closest_waypoint = car_wp_idx


            ## Find the closest visible traffic light (if one exists).
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                ## Get the stop line waypoint index.
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                ## Find closest stop line waypoint index.
                d = temp_wp_idx - car_wp_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        ##self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def process_traffic_lights_collect_data(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint inded
                d = temp_wp_idx - car_wp_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        close_enough = False
        if closest_light:
            if diff < 250 and diff > 0:
                close_enough = True
                ## This should return either the current state of the nearest light.
                state = self.get_light_state_collect_data(light)
            else:
                ## If the light is too far away then return UNKNOWN
                 state = TrafficLight.UNKNOWN
            ##state = self.get_light_state(closest_light)
            ##return line_wp_idx, state
            return state, close_enough

        ##self.waypoints = None
        ##return -1, TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN, close_enough

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')