import json
import torch


class InferModule:
    def __init__(self, model):
        self.model = model
        with open("./model/num2content.json", "r") as f:
            self.num2content = json.load(f)

    def start_inferring(self, question_answer):
        mbti_score = torch.Tensor(
            [question_answer[idx : idx + 3].count(2) for idx in range(0, 12, 3)]
        )

        self.model.eval()
        pred = self.model(mbti_score)

        res = [
            self.num2content[str(idx)] for idx in torch.topk(pred, 3).indices.tolist()
        ]
        mbti_score = list(map(int, mbti_score.tolist()))

        inferringResponse = {
            "hobbyType": {
                "mbtiType": self.get_u_type(mbti_score),
                "scores": [
                    {"scoreE": mbti_score[0]},
                    {"scoreN": mbti_score[1]},
                    {"scoreF": mbti_score[2]},
                    {"scoreJ": mbti_score[3]},
                ],
            },
            "hobbies": [
                {"name": res[0]},
                {"name": res[1]},
                {"name": res[2]},
            ],
        }

        return inferringResponse

    def get_u_type(mbti_score):
        u_type = ""
        if mbti_score[0] >= 2:
            u_type += "E"
        else:
            u_type += "I"
        if mbti_score[1] >= 2:
            u_type += "N"
        else:
            u_type += "S"
        if mbti_score[2] >= 2:
            u_type += "F"
        else:
            u_type += "T"
        if mbti_score[3] >= 2:
            u_type += "J"
        else:
            u_type += "P"

        return u_type
