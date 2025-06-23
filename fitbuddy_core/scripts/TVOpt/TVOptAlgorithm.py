from abc import ABC, abstractmethod


class TVOptAlgorithm(ABC):
    @abstractmethod
    def update_law(self, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        pass

    @abstractmethod
    def dynamics(self, X, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        pass

    @abstractmethod
    def state(self, *args, **kwargs):
        pass
