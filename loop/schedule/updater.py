class ParameterUpdater:

    def __init__(self, schedule, params, opt=None):
        self.schedule = schedule
        self.params = params
        self.opt = opt
        self.start_parameters = None

    def set_optimizer(self, opt):
        self.opt = opt

    def save_start_values(self):
        start = []
        for group in self.opt.param_groups:
            params = {}
            for item in self.params:
                name = item['name']
                if name in group:
                    params[name] = group[name]
            start.append(params)
        self.start_parameters = start

    def current_values(self):
        return [
            {conf['name']: group[conf['name']]
             for conf in self.params}
            for group in self.opt.param_groups]

    def step(self):
        mult = self.schedule.update()
        for i, group in enumerate(self.opt.param_groups):
            for item in self.params:
                name = item['name']
                if name in group:
                    params = self.start_parameters[i]
                    inverse = item.get('inverse', False)
                    start_value = params.get(name)
                    group[name] = start_value * ((1 - mult) if inverse else mult)
