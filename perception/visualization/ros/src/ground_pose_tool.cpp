#include "ground_pick.h"
#include <memory>
#include <mutex>
#include <string>
#include <QKeyEvent>
#include <QTimer>
#include <QApplication>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QShortcut>
#include <QTextEdit>
#include <QAbstractSpinBox>
#include <QVBoxLayout>
#include <OgreCamera.h>
#include <OgreRay.h>
#include <OgreSceneNode.h>
#include <OgreViewport.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <rviz/display_context.h>
#include <rviz/ogre_helpers/arrow.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/string_property.h>
#include <rviz/tool.h>
#include <rviz/panel.h>
#include <rviz/tool_manager.h>
#include <rviz/viewport_mouse_event.h>

namespace social_nav_rviz {
class GroundPoseTool : public rviz::Tool {
 public:
  explicit GroundPoseTool(bool initial) : initial_(initial) {
    shortcut_key_=initial ? 'p' : 'g';
    topic_=new rviz::StringProperty("Topic",initial ? "/social_nav_interactive/initial_pose_candidate" : "/move_base_simple/goal",
        "Pose intent output; this tool never commands the chassis.",getPropertyContainer());
    ground_topic_=new rviz::StringProperty("Ground topic","/social_nav_interactive/ground_map",
        "Supported static terrain cells in the RViz fixed frame.",getPropertyContainer());
    resolution_=new rviz::FloatProperty("Cell size",.1,"Terrain grid resolution in metres.",getPropertyContainer());
    resolution_->setMin(.001);
  }
  ~GroundPoseTool() override { subscriber_.shutdown(); }
  void onInitialize() override {
    setName(initial_ ? "Ground Pose Estimate" : "Ground Nav Goal");
    arrow_.reset(new rviz::Arrow(scene_manager_,nullptr,.6,.06,.2,.12));
    arrow_->setColor(initial_ ? .2 : 1.,initial_ ? 1. : .3,initial_ ? .2 : 1.,1.);
    arrow_->getSceneNode()->setVisible(false);
    subscribe();
  }
  void activate() override {
    deactivate();subscribe();advertise();
    setStatus("Press on supported ground, drag for heading, release to submit. Esc cancels.");
  }
  void deactivate() override {
    dragging_=false;
    if (arrow_) arrow_->getSceneNode()->setVisible(false);
  }
  int processKeyEvent(QKeyEvent* event,rviz::RenderPanel*) override {
    if (event->key()!=Qt::Key_Escape) return 0;
    deactivate();return Render|Finished;
  }
  int processMouseEvent(rviz::ViewportMouseEvent& event) override {
    const auto ray=event.viewport->getCamera()->getCameraToViewportRay(
        double(event.x)/event.viewport->getActualWidth(),double(event.y)/event.viewport->getActualHeight());
    const auto o=ray.getOrigin(),d=ray.getDirection();
    const Point origin{o.x,o.y,o.z},direction{d.x,d.y,d.z};
    const std::string frame=context_->getFixedFrame().toStdString();
    if (event.leftDown()) {
      std::shared_ptr<const Terrain> terrain;
      { std::lock_guard<std::mutex> lock(mutex_);terrain=terrain_; }
      if (!terrain || terrain->frame!=frame ||
          !ground_hit(terrain->cells,resolution_->getFloat(),origin,direction,start_)) {
        deactivate();setStatus("No supported ground here (or map frame mismatch). Nothing submitted.");return Render;
      }
      frame_=frame;dragging_=true;
      arrow_->setPosition(Ogre::Vector3(start_.x,start_.y,start_.z+.05));
      setStatus(QString("Ground (%1, %2, %3). Drag at least 5 cm for heading.")
          .arg(start_.x,0,'f',3).arg(start_.y,0,'f',3).arg(start_.z,0,'f',3));
      return Render;
    }
    if (!dragging_) return 0;
    if (frame!=frame_) { deactivate();setStatus("Fixed frame changed; select again.");return Render; }
    if ((event.type==QEvent::MouseMove && event.left()) || event.leftUp()) {
      Point end;
      // Heading uses the horizontal plane through the picked ground, not Z=0
      // or another object's depth. It does not change the selected position.
      const bool valid=plane_hit(origin,direction,start_.z,end) && std::hypot(end.x-start_.x,end.y-start_.y)>=.05;
      arrow_->getSceneNode()->setVisible(valid);
      if (valid) {
        const double yaw=std::atan2(end.y-start_.y,end.x-start_.x);
        arrow_->setOrientation(Ogre::Quaternion(Ogre::Radian(yaw),Ogre::Vector3::UNIT_Z)*
            Ogre::Quaternion(Ogre::Radian(-Ogre::Math::HALF_PI),Ogre::Vector3::UNIT_Y));
        if (event.leftUp()) {
          if (publisher_.getNumSubscribers()==0) {
            deactivate();setStatus("No pose subscriber connected; nothing published. Check the session and try again.");return Render;
          }
          geometry_msgs::PoseStamped pose;pose.header.frame_id=frame_;pose.header.stamp=ros::Time::now();
          pose.pose.position.x=start_.x;pose.pose.position.y=start_.y;pose.pose.position.z=start_.z;
          pose.pose.orientation.z=std::sin(yaw/2);pose.pose.orientation.w=std::cos(yaw/2);
          if (initial_) {
            geometry_msgs::PoseWithCovarianceStamped message;message.header=pose.header;message.pose.pose=pose.pose;
            message.pose.covariance[0]=message.pose.covariance[7]=.25;
            message.pose.covariance[35]=std::pow(Ogre::Math::PI/12.,2);publisher_.publish(message);
          } else publisher_.publish(pose);
          deactivate();setStatus("Pose published. Check the session's accepted intent marker and planning result.");
          return Render|Finished;
        }
      }
      if (event.leftUp()) { deactivate();setStatus("Heading drag too short or unavailable; nothing submitted."); }
      return Render;
    }
    return 0;
  }
 private:
  struct Terrain { std::string frame;std::vector<Point> cells; };
  void advertise() {
    const auto topic=topic_->getStdString();
    if (publisher_ && topic==published_) return;
    publisher_.shutdown();
    if (initial_) publisher_=node_.advertise<geometry_msgs::PoseWithCovarianceStamped>(topic,1);
    else publisher_=node_.advertise<geometry_msgs::PoseStamped>(topic,1);
    published_=topic;
  }
  void subscribe() {
    const auto topic=ground_topic_->getStdString();
    if (topic==subscribed_) return;
    subscribed_=topic;
    { std::lock_guard<std::mutex> lock(mutex_);terrain_.reset(); }
    subscriber_=node_.subscribe<sensor_msgs::PointCloud2>(topic,1,[this](const sensor_msgs::PointCloud2::ConstPtr& message) {
      auto terrain=std::make_shared<Terrain>();terrain->frame=message->header.frame_id;
      try {
        for (const auto* name : {"x","y","z"}) {
          bool valid=false;
          for (const auto& field : message->fields)
            if (field.name==name && field.datatype==sensor_msgs::PointField::FLOAT32 &&
                field.count==1 && field.offset+sizeof(float)<=message->point_step) valid=true;
          if (!valid) throw std::runtime_error("Expected FLOAT32 XYZ fields");
        }
        if (message->is_bigendian || message->row_step!=message->width*message->point_step ||
            message->data.size()!=size_t(message->row_step)*message->height)
          throw std::runtime_error("Expected a packed little-endian ground cloud");
        sensor_msgs::PointCloud2ConstIterator<float> x(*message,"x"),y(*message,"y"),z(*message,"z");
        terrain->cells.reserve(message->width*message->height);
        for (;x!=x.end();++x,++y,++z)
          if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) terrain->cells.push_back({*x,*y,*z});
      } catch (const std::runtime_error& error) {
        ROS_WARN_STREAM("Ground pose tool rejected terrain: " << error.what());terrain->cells.clear();
      }
      std::lock_guard<std::mutex> lock(mutex_);terrain_=terrain;
    });
  }
  bool initial_,dragging_=false;
  Point start_{};
  std::string frame_,subscribed_,published_;
  ros::NodeHandle node_;
  ros::Publisher publisher_;
  ros::Subscriber subscriber_;
  rviz::StringProperty *topic_,*ground_topic_;
  rviz::FloatProperty* resolution_;
  std::unique_ptr<rviz::Arrow> arrow_;
  std::mutex mutex_;
  std::shared_ptr<const Terrain> terrain_;
};
class GroundGoalTool : public GroundPoseTool { public:GroundGoalTool():GroundPoseTool(false){} };
class GroundInitialPoseTool : public GroundPoseTool { public:GroundInitialPoseTool():GroundPoseTool(true){} };
class ConfirmInitialPoseTool : public rviz::Tool {
 public:
  void onInitialize() override {
    setName("Confirm Initial Pose");
    publisher_=node_.advertise<std_msgs::Empty>("/social_nav_interactive/confirm_initial_pose",1);
  }
  void activate() override {
    if (publisher_.getNumSubscribers()) {
      publisher_.publish(std_msgs::Empty());
      setStatus("Confirmation requested. Check the session status.");
    } else setStatus("No initial-pose confirmation listener; nothing sent.");
    QTimer::singleShot(0,this,[this]() {
      auto* manager=context_->getToolManager();
      if (manager && manager->getCurrentTool()==this) manager->setCurrentTool(manager->getDefaultTool());
    });
  }
  void deactivate() override {}
 private:
  ros::NodeHandle node_;
  ros::Publisher publisher_;
};

class SessionPanel : public rviz::Panel {
 public:
  explicit SessionPanel(QWidget* parent=nullptr):rviz::Panel(parent) {
    publisher_=node_.advertise<std_msgs::String>("/social_nav_interactive/panel_command",10);
    auto* layout=new QVBoxLayout(this);
    auto* instructions=new QLabel(QString::fromUtf8("先选择实机起点。WASD 沿车体方向平移，Q/E 旋转。\n确认前请检查点云预览。"));
    instructions->setWordWrap(true);layout->addWidget(instructions);
    auto* grid=new QGridLayout();layout->addLayout(grid);
    add(grid,"Q 左转","q",0,0);add(grid,"W 前移","w",0,1);add(grid,"E 右转","e",0,2);
    add(grid,"A 左移","a",1,0);add(grid,"S 后移","s",1,1);add(grid,"D 右移","d",1,2);
    add(grid,"轮椅使能","arm",2,0);add(grid,"取消 / 退出使能","cancel",2,1);
    add(grid,"退出会话","exit",2,2);
    status_=new QLabel(QString::fromUtf8("命令沿用现有会话检查。"));
    status_->setWordWrap(true);layout->addWidget(status_);
    state_=new QLabel(QString::fromUtf8("等待会话状态"));state_->setWordWrap(true);layout->addWidget(state_);
    subscriber_=node_.subscribe<std_msgs::String>("/social_nav_interactive/state",1,
        [this](const std_msgs::String::ConstPtr& message) {
          const auto object=QJsonDocument::fromJson(QByteArray::fromStdString(message->data)).object();
          QMetaObject::invokeMethod(this,[this,object]() {
            if (object.contains("accepted")) {
              status_->setText(object.value("accepted").toBool() ? QString::fromUtf8("会话已接收操作")
                  : QString::fromUtf8("操作未接收：")+object.value("error").toString());
            }
            if (object.value("chassis").isObject()) {
              const auto chassis=object.value("chassis").toObject();
              state_->setText(QString::fromUtf8("底盘状态：")+chassis.value("state").toString(QString::fromUtf8("未知"))+
                  (chassis.value("status_fresh").toBool() ? QString::fromUtf8("（新鲜）") : QString::fromUtf8("（不可用或过期）")));
            }
          },Qt::QueuedConnection);
        });
    for (const auto key : {"w","a","s","d","q","e"}) {
      auto* shortcut=new QShortcut(QKeySequence(QString::fromLatin1(key)),this);
      shortcut->setContext(Qt::WindowShortcut);
      connect(shortcut,&QShortcut::activated,this,[this,key]() {
        auto* focus=QApplication::focusWidget();
        if (qobject_cast<QLineEdit*>(focus) || qobject_cast<QTextEdit*>(focus) ||
            qobject_cast<QPlainTextEdit*>(focus) || qobject_cast<QAbstractSpinBox*>(focus)) return;
        send(key);
      });
    }
  }
 private:
  void add(QGridLayout* grid,const char* title,const char* command,int row,int col) {
    auto* button=new QPushButton(QString::fromUtf8(title));grid->addWidget(button,row,col);
    connect(button,&QPushButton::clicked,this,[this,command](){send(command);});
  }
  void send(const char* command) {
    if (!publisher_.getNumSubscribers()) { status_->setText(QString::fromUtf8("会话不可用，命令未发送。"));return; }
    std_msgs::String message;message.data=command;publisher_.publish(message);
    status_->setText(QString::fromUtf8("已发送 %1，等待会话确认。").arg(command));
  }
  ros::NodeHandle node_;ros::Publisher publisher_;ros::Subscriber subscriber_;QLabel *status_,*state_;
};
} // namespace social_nav_rviz

PLUGINLIB_EXPORT_CLASS(social_nav_rviz::GroundGoalTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::GroundInitialPoseTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::ConfirmInitialPoseTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::SessionPanel,rviz::Panel)
