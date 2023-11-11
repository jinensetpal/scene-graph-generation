#!/usr/bin/env python3

from .. import const
import dagshub
import mlflow

conda_env = {'channels': ['defaults'],
             'dependencies': [
                 'python~=3.11',
                 'pip',
                 {'pip': ['mlflow',
                          'pillow',
                          'torch',
                          'torch_geometric']}],
             'name': 'env'}


class SGGWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        from .arch import SceneGraphGenerator
        import torch

        self.SceneGraphGenerator = SceneGraphGenerator
        self.torch = torch

    def load_context(self, context):
        self.model = self.SceneGraphGenerator()
        self.model.eval()
        self.model.load_state_dict(self.torch.load(context.artifacts["path"]))

    def predict(self, context, model_input):
        with self.torch.no_grad():
            result = self.model(model_input)

        return {"result": result}


if __name__ == '__main__':
    dagshub.init(*const.REPO_NAME.split('/')[::-1])
    model = SGGWrapper()
    artifacts = {'path': str(const.MODEL_DIR / 'model.pt')}

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path='models',
            python_model=model,
            code_path=['src',],
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name='main',
        )
