import aim
import tensorboard
class loggerBase(object):
    def __init__(self,name:str) -> None:
        pass
    def track(self,value,name,step,context=None):
        pass
    def track_figure(self,value,name,step,context=None):
        pass
    def track_image(self,value,name,step,context=None):
        pass

class aim_logger(loggerBase):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.run=aim.Run(experiment=name)

    def track(self, value, name, step, context=None):
        self.run.track(value,name,step=step,context=context)
        return 

    def track_figure(self, value, name, step, context=None):
        self.run.track(aim.Figure(value),name,step=step,context=context)
        return 

    def track_image(self, value, name, step, context=None):
        self.run.track(aim.Image(value),name,step=step,context=context)
        return 