from flask import Flask, jsonify, request

import torch
from model.infer import InferModule
from model.module import MyModule

model = MyModule()
model.load_state_dict(torch.load("./data/model.pt"))
IM = InferModule(model)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def healthCheck():
    return 'Ok', 200

@app.route('/test-responses', methods=['POST'])
def inferHobbiesAndType():
    testResponses: list = request.get_json()
    # 질문 번호순서대로 정렬
    testResponses.sort(key=lambda x: x['questionNumber'])
    #리스트로 변환
    inferringInputData = []
    for testResponse in testResponses:
        inferringInputData.append(testResponse['answerNumber'])
    #추론 딥러닝 모델에 추론 요청
    inferringResponse = IM.start_inferring(inferringInputData)

    return jsonify(inferringResponse), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
