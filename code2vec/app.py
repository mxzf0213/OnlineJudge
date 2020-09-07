from utils.api import APIView
from code2vec.main import Code2tags
from account.decorators import login_required
import os

model = Code2tags(problem_info=False)
modelSaved_path = os.path.join(model.modelSaved_path,"origin_model")
model.load_savedModel(os.path.join(modelSaved_path,"cp-0070.ckpt"))

class code2vecAPI(APIView):
    @login_required
    def get(self, request):
        code = request.GET.get('code')
        results = model.predict_code(code)
        return self.success({'name':results[0],'prob':results[1]})

# if __name__ == '__main__':
#     app.run()
