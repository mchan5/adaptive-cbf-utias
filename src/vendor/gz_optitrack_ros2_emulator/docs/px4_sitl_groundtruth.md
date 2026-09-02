## Add the ground-truth plugin in the gazebo model
- The gz model files are contained in the following directories: ``/PX4_Autopilot/Tools/simulation/gz/models/MODEL_FOLDER/model.sdf``
- To create an odometry publisher on a link in a Gazebo model, in the sdf file of your PX4 SITL model, add the following field to the XML file:
```
<model>
…other stuff in sdf definition
    <!-- Observers -->
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>SUBMODEL_NAME::LINK_NAME</odom_frame>
      <odom_topic>model/<model_name>/groundtruth_odometry</odom_topic>
      <dimensions>3</dimensions>
      <odom_publish_frequency>100</odom_publish_frequency>  <!-- HZ -->
    </plugin>
…other stuff
</model>
</sdf>
```
- The odom_frame field is the name of the link of which the odometry is published.
- The odom_topic field is the gz topic. Set the odometry topic in the following format. (<model_name> is the name of the link). <model_name> must match the name listed in the params.yaml.