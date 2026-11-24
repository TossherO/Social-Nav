#include "ground_pick.h"
#include <memory>
#include <mutex>
#include <string>
#include <QKeyEvent>
#include <QTimer>
#include <QApplication>
#include <QButtonGroup>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QStringList>
#include <QTextEdit>
#include <QAbstractSpinBox>
#include <QUuid>
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
#include <rviz/display_group.h>
#include <rviz/ogre_helpers/arrow.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/string_property.h>
#include <rviz/tool.h>
#include <rviz/panel.h>
#include <rviz/tool_manager.h>
#include <rviz/viewport_mouse_event.h>
#include <rviz/visualization_manager.h>

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
    auto* grid=new QGridLayout();layout->addLayout(grid);
    add(grid,"Q 左转","q","adjust",0,0);add(grid,"W 前移","w","adjust",0,1);add(grid,"E 右转","e","adjust",0,2);
    add(grid,"A 左移","a","adjust",1,0);add(grid,"S 后移","s","adjust",1,1);add(grid,"D 右移","d","adjust",1,2);
    auto* confirm=add(grid,"确认实机起点","confirm","confirm",2,0);
    confirm->setToolTip(QString::fromUtf8("请先将点云预览与静态地图对齐。"));
    add(grid,"暂停回放","pause","pause",2,1);add(grid,"继续回放","resume","resume",2,2);
    add(grid,"启动影子导航","shadow_start","shadow_start",3,0);
    add(grid,"停止影子导航","shadow_stop","shadow_stop",3,1);
    add(grid,"轮椅使能","arm","arm",4,0);add(grid,"取消目标 / 解除使能","cancel","cancel",4,1);
    add(grid,"退出会话","exit","exit",4,2);

    auto* maps=new QGridLayout();layout->addLayout(maps);
    static_button_=new QPushButton(QString::fromUtf8("静态底图"));
    navigation_button_=new QPushButton(QString::fromUtf8("导航底图"));
    search_button_=new QPushButton(QString::fromUtf8("搜索快照"));
    map_buttons_=new QButtonGroup(this);
    for (auto* button : {static_button_,navigation_button_,search_button_}) {
      button->setCheckable(true);
      button->setEnabled(false);
      map_buttons_->addButton(button);
    }
    maps->addWidget(static_button_,0,0);maps->addWidget(navigation_button_,0,1);
    maps->addWidget(search_button_,0,2);
    connect(static_button_,&QPushButton::clicked,this,[this]() {selectBasemap(MapView::Static);});
    connect(navigation_button_,&QPushButton::clicked,this,[this]() {selectBasemap(MapView::Navigation);});
    connect(search_button_,&QPushButton::clicked,this,[this]() {selectBasemap(MapView::Search);});

    map_legend_=new QLabel(this);
    map_legend_->setTextFormat(Qt::RichText);
    map_legend_->setWordWrap(true);
    map_legend_->hide();
    layout->addWidget(map_legend_);

    auto* status_scroll=new QScrollArea(this);
    status_scroll->setFrameShape(QFrame::NoFrame);
    status_scroll->setWidgetResizable(true);
    auto* status_panel=new QWidget();
    auto* status_layout=new QVBoxLayout(status_panel);
    status_layout->setContentsMargins(0,0,0,0);
    status_scroll->setWidget(status_panel);
    layout->addWidget(status_scroll,1);

    map_source_=new QLabel();map_source_->setTextFormat(Qt::PlainText);
    map_source_->setWordWrap(true);status_layout->addWidget(map_source_);

    status_=new QLabel(QString::fromUtf8("交互会话未连接"));
    status_->setTextFormat(Qt::PlainText);
    status_->setWordWrap(true);status_layout->addWidget(status_);
    playback_=new QLabel();playback_->setWordWrap(true);status_layout->addWidget(playback_);
    shadow_=new QLabel();shadow_->setWordWrap(true);status_layout->addWidget(shadow_);
    chassis_=new QLabel();chassis_->setWordWrap(true);status_layout->addWidget(chassis_);
    status_layout->addStretch(1);
    panel_subscriber_=node_.subscribe<std_msgs::String>("/social_nav_interactive/panel_state",1,
        [this](const std_msgs::String::ConstPtr& message) {
          const auto object=QJsonDocument::fromJson(QByteArray::fromStdString(message->data)).object();
          QMetaObject::invokeMethod(this,[this,object]() {updateSession(object);},Qt::QueuedConnection);
        });
    reply_subscriber_=node_.subscribe<std_msgs::String>("/social_nav_interactive/panel_reply",10,
        [this](const std_msgs::String::ConstPtr& message) {
          const auto object=QJsonDocument::fromJson(QByteArray::fromStdString(message->data)).object();
          QMetaObject::invokeMethod(this,[this,object]() {updateReply(object);},Qt::QueuedConnection);
        });
    basemap_subscriber_=node_.subscribe<std_msgs::String>("/social_nav_runtime_view/basemap_source",1,
        [this](const std_msgs::String::ConstPtr& message) {
          const auto source=QJsonDocument::fromJson(QByteArray::fromStdString(message->data)).object();
          QMetaObject::invokeMethod(this,[this,source]() {basemap_source_=source;updateBasemapControls();},
                                    Qt::QueuedConnection);
        });
    for (const auto key : {"w","a","s","d","q","e"}) {
      auto* shortcut=new QShortcut(QKeySequence(QString::fromLatin1(key)),this);
      shortcut->setContext(Qt::WindowShortcut);
      shortcut->setEnabled(false);
      adjustment_shortcuts_.append(shortcut);
      connect(shortcut,&QShortcut::activated,this,[this,key]() {
        auto* focus=QApplication::focusWidget();
        if (qobject_cast<QLineEdit*>(focus) || qobject_cast<QTextEdit*>(focus) ||
            qobject_cast<QPlainTextEdit*>(focus) || qobject_cast<QAbstractSpinBox*>(focus)) return;
        if (controls_.value("adjust").toBool() && publisher_.getNumSubscribers()) send(key);
      });
    }
    auto* timer=new QTimer(this);
    connect(timer,&QTimer::timeout,this,[this]() {updateButtons();updateBasemapControls();});
    timer->start(250);
  }
  ~SessionPanel() override {
    panel_subscriber_.shutdown();
    reply_subscriber_.shutdown();
    basemap_subscriber_.shutdown();
  }
  void onInitialize() override {
    manager_ready_=true;
    QTimer::singleShot(0,this,[this]() {updateBasemapControls();});
  }
 private:
  enum class MapView { Static, Navigation, Search };

  QPushButton* add(QGridLayout* grid,const char* title,const char* command,const char* control,int row,int col) {
    auto* button=new QPushButton(QString::fromUtf8(title));grid->addWidget(button,row,col);
    buttons_.append({button,QString::fromLatin1(control)});
    button->setEnabled(false);
    connect(button,&QPushButton::clicked,this,[this,command](){send(command);});
    return button;
  }
  void send(const char* command) {
    if (!publisher_.getNumSubscribers()) { status_->setText(QString::fromUtf8("会话不可用，命令未发送。"));return; }
    const auto request_id=QUuid::createUuid().toString(QUuid::WithoutBraces);
    pending_request_=request_id;
    const QJsonObject request{{"command",QString::fromLatin1(command)},{"request_id",request_id}};
    std_msgs::String message;
    message.data=QJsonDocument(request).toJson(QJsonDocument::Compact).toStdString();
    publisher_.publish(message);
    status_->setText(QString::fromUtf8("已发送 %1，等待会话确认。").arg(command));
    QTimer::singleShot(8000,this,[this,request_id]() {
      if (pending_request_!=request_id) return;
      pending_request_.clear();
      status_->setText(QString::fromUtf8("会话回执超时，操作结果未知。"));
    });
  }
  void updateReply(const QJsonObject& object) {
    if (pending_request_.isEmpty() || object.value("request_id").toString()!=pending_request_ ||
        !object.value("accepted").isBool()) return;
    pending_request_.clear();
    const auto error=object.value("error").toString();
    if (object.value("accepted").toBool()) status_->setText(QString::fromUtf8("会话已接收操作"));
    else if (error.isEmpty()) status_->setText(QString::fromUtf8("会话拒绝了操作"));
    else status_->setText(QString::fromUtf8("操作未接收：")+error);
  }
  void updateButtons() {
    const bool connected=publisher_.getNumSubscribers()>0;
    if (!connected && connected_) {
      controls_={};
      pending_request_.clear();
      status_->setText(QString::fromUtf8("交互会话未连接"));
      playback_->clear();
      shadow_->clear();
      shadow_->setToolTip(QString());
      chassis_->clear();
    }
    if (connected && !connected_) status_->setText(QString::fromUtf8("交互会话已连接"));
    connected_=connected;
    for (const auto& entry : buttons_) entry.first->setEnabled(connected && controls_.value(entry.second).toBool());
    for (auto* shortcut : adjustment_shortcuts_) shortcut->setEnabled(connected && controls_.value("adjust").toBool());
  }
  void updateSession(const QJsonObject& object) {
    if (!object.value("controls").isObject()) return;
    controls_=object.value("controls").toObject();
    const auto playback=object.value("playback_state").toString();
    if (playback=="ended") playback_->setText(QString::fromUtf8("回放已结束，当前画面可用于检查"));
    else if (playback=="paused") playback_->setText(QString::fromUtf8("回放已暂停"));
    else if (playback=="running") playback_->setText(QString::fromUtf8("正在回放"));
    else playback_->clear();
    updateShadow(object.value("shadow_navigation").toObject());
    if (!object.value("hardware_enabled").toBool()) chassis_->clear();
    else {
      const auto chassis=object.value("chassis").toObject();
      chassis_->setText(QString::fromUtf8("底盘状态：")+chassis.value("state").toString(QString::fromUtf8("未知"))+
          (chassis.value("status_fresh").toBool() ? QString::fromUtf8("（状态有效）") : QString::fromUtf8("（不可用或过期）")));
    }
    updateButtons();
  }
  void updateShadow(const QJsonObject& shadow) {
    const auto shadow_state=shadow.value("state").toString();
    QString tooltip=shadow.value("reason").toString();
    if (shadow_state=="disabled") shadow_->setText(QString::fromUtf8("影子导航未启动"));
    else if (shadow_state=="preparing") shadow_->setText(QString::fromUtf8("正在准备影子导航"));
    else if (shadow_state=="waiting") shadow_->setText(QString::fromUtf8("影子导航等待前方目标"));
    else if (shadow_state=="unavailable") shadow_->setText(QString::fromUtf8("影子导航不可用"));
    else if (shadow_state=="planning") {
      const auto planning_status=shadow.value("planning_status").toString();
      const auto planning_reason=shadow.value("planning_reason").toString();
      const auto dwa_status=shadow.value("dwa_status").toString();
      const auto dwa_reason=shadow.value("dwa_reason").toString();

      QString reference=QString::fromUtf8("计算中");
      if (planning_status=="reference_ready") reference=QString::fromUtf8("已生成");
      else if (planning_reason=="goal_occupied" || planning_reason=="goal_footprint_occupied")
        reference=QString::fromUtf8("目标位置被占用");
      else if (planning_status=="unavailable" && planning_reason!="waiting_for_background_reference")
        reference=QString::fromUtf8("未生成");

      QString dwa=QString::fromUtf8("等待计算");
      if (dwa_status=="candidate") dwa=QString::fromUtf8("已生成候选轨迹");
      else if (dwa_reason=="reference_has_no_path") dwa=QString::fromUtf8("缺少可用参考路径");
      else if (dwa_status=="unavailable" || !dwa_reason.isEmpty()) dwa=QString::fromUtf8("未生成");
      shadow_->setText(QString::fromUtf8("影子导航\n参考路径：%1\nDWA：%2").arg(reference,dwa));

      QStringList reasons;
      if (!tooltip.isEmpty()) reasons.append(tooltip);
      if (!planning_reason.isEmpty()) reasons.append(QString::fromUtf8("参考路径：")+planning_reason);
      if (!dwa_reason.isEmpty()) reasons.append(QString::fromUtf8("DWA：")+dwa_reason);
      tooltip=reasons.join('\n');
    }
    else shadow_->clear();
    shadow_->setToolTip(tooltip);
  }
  rviz::Display* findDisplay(const QString& name) const {
    if (!manager_ready_) return nullptr;
    auto* root=vis_manager_->getRootDisplayGroup();
    for (int i=0;i<root->numDisplays();++i) {
      auto* display=root->getDisplayAt(i);
      if (display->getName()==name) return display;
    }
    return nullptr;
  }
  void selectBasemap(MapView view) {
    if (auto* display=findDisplay("Static map")) display->setEnabled(view!=MapView::Search);
    if (auto* display=findDisplay("Navigation basemap")) display->setEnabled(view==MapView::Navigation);
    if (auto* display=findDisplay("Navigation window")) display->setEnabled(view==MapView::Navigation);
    if (auto* display=findDisplay("Reference search snapshot")) display->setEnabled(view==MapView::Search);
  }
  void updateMapLegend(bool show_static,bool show_navigation,bool show_window,bool show_search) {
    // Swatches match the RViz point cloud palettes in visualization/surface.py.
    const auto swatch=[](const char* color,const char* label) {
      return QString::fromLatin1("<span style=\"color:%1\">&#9632;</span>&nbsp;%2")
          .arg(QString::fromLatin1(color),QString::fromUtf8(label));
    };
    QStringList entries;
    if (show_search) {
      entries << swatch("#9EA9B4","搜索空闲") << swatch("#CF5D53","搜索占据") << swatch("#CB974A","条件代价");
    } else {
      if (show_static) {
        entries << swatch("#9EA9B4","静态空闲") << swatch("#CF5D53","静态占据")
                << swatch("#707885","静态未知") << swatch("#72A0B7","估计地面");
      }
      if (show_navigation) {
        entries << swatch("#72C8C3","局部空闲") << swatch("#F47B63","局部占据");
      }
      if (show_window) entries << swatch("#59FFDB","局部范围");
    }
    const auto legend=entries.join(" &nbsp; ");
    if (map_legend_->text()!=legend) map_legend_->setText(legend);
    map_legend_->setVisible(!entries.isEmpty());
  }
  void updateBasemapControls() {
    auto* static_map=findDisplay("Static map");
    auto* navigation=findDisplay("Navigation basemap");
    auto* window=findDisplay("Navigation window");
    auto* search=findDisplay("Reference search snapshot");
    if (!basemap_subscriber_.getNumPublishers()) basemap_source_={};
    const auto fixed_frame=basemap_source_.value("static_frame").toString();
    if (static_map && !initial_map_frame_seen_ && !fixed_frame.isEmpty()) {
      initial_map_frame_seen_=true;
      // The standalone view starts in odom for mapless runs. Select the asset frame
      // before localization and reconnect latched layers cleared by the frame change.
      if (vis_manager_->getFixedFrame()=="odom" && fixed_frame!="odom") {
        vis_manager_->setFixedFrame(fixed_frame);
        for (auto* display : {static_map,navigation,window,search}) {
          if (display && display->isEnabled()) {
            display->setEnabled(false);
            display->setEnabled(true);
          }
        }
      }
    }
    const auto query=basemap_source_.value("search_query_id");
    const bool search_available=query.isDouble();
    if (search && search->isEnabled() && !search_available) selectBasemap(MapView::Navigation);
    static_button_->setEnabled(static_map!=nullptr);
    navigation_button_->setEnabled(navigation && window);
    search_button_->setEnabled(search && search_available);
    const bool show_search=search && search->isEnabled();
    const bool show_navigation=(navigation && navigation->isEnabled()) || (window && window->isEnabled());
    updateMapLegend(static_map && static_map->isEnabled(),navigation && navigation->isEnabled(),
                    window && window->isEnabled(),show_search);
    map_buttons_->setExclusive(false);
    search_button_->setChecked(show_search);
    navigation_button_->setChecked(!show_search && show_navigation);
    static_button_->setChecked(!show_search && !show_navigation && static_map && static_map->isEnabled());
    map_buttons_->setExclusive(true);

    QStringList layers;
    if (!show_search && static_map && static_map->isEnabled()) {
      const auto topic=static_map->subProp("Topic")->getValue().toString();
      if (topic=="/social_nav_interactive/static_basemap" || basemap_source_.value("static_available").toBool())
        layers.append(QString::fromUtf8("静态地图"));
    }
    if (!show_search && show_navigation && basemap_source_.value("local_available").toBool())
      layers.append(basemap_source_.value("frozen").toBool()
          ? QString::fromUtf8("暂停时的局部导航图") : QString::fromUtf8("局部导航图"));
    const auto search_status=search_available
        ? QString::fromUtf8("搜索快照可用，查询 #%1").arg(query.toVariant().toLongLong())
        : QString::fromUtf8("搜索快照不可用");
    if (!static_map && !navigation && !window && !search) map_source_->clear();
    else if (show_search) map_source_->setText(search_status);
    else map_source_->setText((layers.isEmpty() ? QString::fromUtf8("底图不可用") : layers.join(" + "))+
                              "\n"+search_status);
  }
  ros::NodeHandle node_;
  ros::Publisher publisher_;
  ros::Subscriber panel_subscriber_,reply_subscriber_,basemap_subscriber_;
  QJsonObject controls_,basemap_source_;
  QList<QPair<QPushButton*,QString>> buttons_;
  QList<QShortcut*> adjustment_shortcuts_;
  QButtonGroup* map_buttons_;
  QPushButton *static_button_,*navigation_button_,*search_button_;
  QLabel *status_,*playback_,*shadow_,*chassis_,*map_source_,*map_legend_;
  QString pending_request_;
  bool connected_=false;
  bool manager_ready_=false;
  bool initial_map_frame_seen_=false;
};
} // namespace social_nav_rviz

PLUGINLIB_EXPORT_CLASS(social_nav_rviz::GroundGoalTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::GroundInitialPoseTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::ConfirmInitialPoseTool,rviz::Tool)
PLUGINLIB_EXPORT_CLASS(social_nav_rviz::SessionPanel,rviz::Panel)

// Startup checks use this symbol to detect stale RViz plugins.
extern "C" int social_nav_rviz_session_panel_abi() { return 4; }
