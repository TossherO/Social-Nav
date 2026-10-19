#! /usr/bin/env python
# -*- coding: UTF-8 -*-

# conclusion: 
# 1. ROS is synchronized with datetime in python
# 2. datetime in python is synchronized with system time
# but ... I saw it is not sometimes (test it in the dataset wheelchair, which is offline sometimes)
import rospy
import datetime
 
if __name__ == "__main__":
    # 初始化节点
    rospy.init_node("time_p")
    # 调用rospy库中的Time类中的now()函数.查询当前时间,参考系为1970年1月1日00:00.
    # @return: L{Time} instance for current time,返回当前时间的实例.
    # 打印当前时间,利用to_sec()将得到的ros时间转化为浮点型
    while rospy.is_shutdown()==False:
        # time_now = rospy.Time.now()
        time_now = rospy.get_rostime()
        print("当前ROS时间为: %.2f"%(time_now.to_sec()))
        # rospy.loginfo("当前ros时间: %.2f ",time_now.to_sec())
        print("当前系统时间: "+ str(datetime.datetime.now()))
    # # 指定时间,参考系为1970年1月1日00:00.
    # # 方法一:直接传入浮点数
    # time1 = rospy.Time(100.5)
    # # 方法二:传入秒与纳秒,通过逗号分隔.
    # time2 = rospy.Time(100,123456789)
    # # 打印两个方法指定的时刻.
    # rospy.loginfo("指定时刻1为:%.2f", time1.to_sec())
