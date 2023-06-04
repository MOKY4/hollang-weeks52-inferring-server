### Usage

```Python
import torch
from model.infer import InferModule
from model.module import MyModule

model = MyModule()
model.load_state_dict(torch.load("./data/model.pt"))
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    result = IM.start_inferring([1,1,2,2,1,2,1,2,1,2,1,2])
    print(result)

    '''
    {
        'hobbyType': {
            'name': 'INTJ',
            'scores': [
                {'scoreE': 1},
                {'scoreN': 2},
                {'scoreF': 1},
                {'scoreJ': 2}
            ]
        },
        'hobbies': [
            {'name': '일러스트 정복하기'},
            {'name': '인스타툰 제작'},
            {'name': '카피라이팅 실습'}
        ]
    }
    '''
```
