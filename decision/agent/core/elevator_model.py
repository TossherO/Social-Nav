"""Transient image-only model judgment while map localization is unavailable."""
import hashlib

from .media import image_part
from .model import Model
from .store import Store,encode,shared_budget


SYSTEM=('你在人工驾驶的轮椅任务中观察乘电梯过程。根据当前相机图像及按时间排列的历史观察，'
        '判断轮椅当前位于电梯轿厢内、轿厢外还是无法判断。先进入轿厢，之后离开轿厢；'
        '画面不足时选择 unknown。不要根据任务目标、等待时间或楼层编号猜测结果；'
        '不要推断所处楼层，也不要指挥轮椅。调用 record_elevator_observation，简述可见依据，'
        '并决定是否保留本张图像供本次乘梯后续观察；乘梯结束后临时观察都会清除。')


def decide_elevator(config, session, transition, history, frame, *, cache_mode='implicit', thinking_mode='off', transport=None):
    if transition['status'] not in ('awaiting_entry','awaiting_exit') or len(history)>4:
        raise ValueError('Elevator observation is not expected')
    content=[dict(type='text',text=encode(dict(source='elevator_monitor',status=transition['status'],
                                               instruction='历史观察依时间排列；最后一张为当前图像。')))]
    previous=-1
    for item in history:
        record=item['record'];camera=record['camera'];image=item['image']
        if (camera['sequence']<=previous or hashlib.sha256(image).hexdigest()!=record['image_sha256']):
            raise ValueError('Invalid elevator history image identity')
        previous=camera['sequence']
        content.extend([dict(type='text',text=encode(dict(source='historical_elevator_image',
            stamp_s=camera['stamp_s'],judgment=record['judgment'],evidence=record['evidence']))),
            image_part(image,768,'JPEG')])
    current=frame['record'];image=frame['image']
    if current['sequence']<=previous or current['source']!='ros_camera_independent_of_localization':
        raise ValueError('Current elevator image must follow its history')
    content.extend([dict(type='text',text=encode(dict(source='current_elevator_image',stamp_s=current['stamp_s']))),
                    image_part(image,768,'JPEG')])
    store,budget=Store(session),shared_budget()
    try:
        model=Model(config,budget,store,transport=transport,cache_mode=cache_mode,thinking_mode=thinking_mode)
        reply=model.request(SYSTEM,[dict(role='user',content=content)],tool_names=('record_elevator_observation',),
                            tool_choice='required',transient=True)
        if reply.action is None or reply.action['tool']!='record_elevator_observation':
            raise ValueError('Elevator observation requires its native tool response')
        return reply.action['arguments']
    finally:
        budget.close();store.close()
