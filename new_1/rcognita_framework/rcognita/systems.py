"""
This module contains a generic interface for systems (environments) as well as concrete systems as realizations of the former

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from calendar import c
import numpy as np
from numpy.random import randn
from .utilities import rc
from scipy.integrate import odeint

import pandas as pd

sin = np.sin
cos = np.cos
pi  = np.pi
sqrt = np.sqrt

class System:
    """
     Interface class of dynamical systems a.k.a. environments.
     Concrete systems should be built upon this class.
     To design a concrete system: inherit this class, override:
         | :func:`~systems.system._compute_state_dynamics` :
         | right-hand side of system description (required)
         | :func:`~systems.system._compute_disturbance_dynamics` :
         | right-hand side of disturbance model (if necessary)
         | :func:`~systems.system._dynamic_control` :
         | right-hand side of controller dynamical model (if necessary)
         | :func:`~systems.system.out` :
         | system out (if not overridden, output is identical to state)

     Attributes
     ----------
     sys_type : : string
         Type of system by description:

         | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
         | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
         | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`

     where:

         | :math:`state` : state
         | :math:`action` : input
         | :math:`disturb` : disturbance

     The time variable ``time`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
     For the latter case, however, you already have the input and disturbance at your disposal.

     Parameters of the system are contained in ``pars`` attribute.

     dim_state, dim_input, dim_output, dim_disturb : : integer
         System dimensions
     pars : : list
         List of fixed parameters of the system
     action_bounds : : array of shape ``[dim_input, 2]``
         Box control constraints.
         First element in each row is the lower bound, the second - the upper bound.
         If empty, control is unconstrained (default)
     is_dynamic_controller : : 0 or 1
         If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
     is_disturb : : 0 or 1
         If 0, no disturbance is fed into the system
     pars_disturb : : list
         Parameters of the disturbance model

    Each concrete system must realize ``System`` and define ``name`` attribute.

    """

    def __init__(
        self,
        sys_type,
        dim_state,
        dim_input,
        dim_output,
        dim_disturb,
        pars=[],
        is_dynamic_controller=0,
        is_disturb=0,
        pars_disturb=[],
    ):

        """
        Parameters
        ----------
        sys_type : : string
            Type of system by description:

            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`

        where:

            | :math:`state` : state
            | :math:`action` : input
            | :math:`disturb` : disturbance

        The time variable ``time`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
        For the latter case, however, you already have the input and disturbance at your disposal.

        Parameters of the system are contained in ``pars`` attribute.

        dim_state, dim_input, dim_output, dim_disturb : : integer
            System dimensions
        pars : : list
            List of fixed parameters of the system
        action_bounds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default)
        is_dynamic_controller : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
        is_disturb : : 0 or 1
            If 0, no disturbance is fed into the system
        pars_disturb : : list
            Parameters of the disturbance model
        """

        self.sys_type = sys_type

        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb
        self.pars = pars
        self.is_dynamic_controller = is_dynamic_controller
        self.is_disturb = is_disturb
        self.pars_disturb = pars_disturb

        # Track system's state
        self._state = np.zeros(dim_state)

        # Current input (a.k.a. action)
        self.action = np.zeros(dim_input)

        if is_dynamic_controller:
            if is_disturb:
                self._dim_full_state = (
                    self.dim_state + self.dim_disturb + self.dim_input
                )
            else:
                self._dim_full_state = self.dim_state
        else:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb
            else:
                self._dim_full_state = self.dim_state

    def _compute_state_dynamics(self, time, state, action, disturb):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, ``_compute_state_dynamics`` should return a number in :math:`[0,1]`

        """
        pass

    def _compute_disturbance_dynamics(self, time, disturb):
        """
        Dynamical disturbance model depending on the system type:

        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D disturb = f_q(disturb)`
        | ``sys_type = "discr_fnc"`` : :math:`disturb^+ = f_q(disturb)`
        | ``sys_type = "discr_prob"`` : :math:`disturb^+ \sim P_Q(disturb^+|disturb)`

        """
        pass

    def _dynamic_control(self, time, action, observation):
        """
        Right-hand side of a dynamical controller. When ``is_dynamic_control=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.

        Depending on the system type, can be:

        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D action = f_u(action, observation)`
        | ``sys_type = "discr_fnc"`` : :math:`action^+ = f_u(action, observation)`
        | ``sys_type = "discr_prob"`` : :math:`action^+ \sim P_U(action^+|action, observation)`

        """
        Daction = rc.zeros(self.dim_input)

        return Daction

    def out(self, state, time=None, action=None):

        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``state`` since no physical processes transmit input to output instantly.

        See also
        --------
        :func:`~systems.system._compute_state_dynamics`

        """
        # Trivial case: output identical to state
        observation = state
        return observation

    def receive_action(self, action):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~systems.system.out`.

        Parameters
        ----------
        action : : array of shape ``[dim_input, ]``
            Action

        """
        self.action = action

    def compute_closed_loop_rhs(self, time, state_full):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.

        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state

        """

        rhs_full_state = np.zeros(self._dim_full_state)

        state = state_full[0 : self.dim_state]
#         print('state', state)
#         print('self.action', self.action)

        if self.is_disturb:
            disturb = state_full[self.dim_state :]
        else:
            disturb = []

        
        if self.is_dynamic_controller:
            action = state_full[-self.dim_input :]
            observation = self.out(state)
            rhs_full_state[-self.dim_input :] = self._ctrlDyn(time, action, observation)
        else:
            # Fetch the control action stored in the system
            action = self.action
            
   
        rhs_full_state[0 : self.dim_state] = self._compute_state_dynamics(
            time, state, action, disturb
        )

        if self.is_disturb:
            rhs_full_state[self.dim_state :] = self._compute_disturbance_dynamics(
                time, disturb
            )

        # Track system's state
        self._state = state

        return rhs_full_state

class TWR(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #robots constants
        self.lin_exp = -2
        self.integral_alpha = 0
        self.R  = 10.75 * 10**self.lin_exp      #wheel radius cm
        self.W  = 37.95 * 10**self.lin_exp      #distance between wheels cm
        self.ro = self.W / 2      #half of the distance between wheels 
        self.z  = 33.05 * 10**self.lin_exp      #heigh of the center of mass cm
        self.L  = self.z - self.R      #stick

        self.I_w   = 80    * 10**(2*self.lin_exp)      #wheel's momtent of inertial n/cm2
        self.I_z   = 3305  * 10**(2*self.lin_exp)     #whole robots's moment of inertia around vetrical axis n/cm2
        self.I_psi = 4762  * 10**(2*self.lin_exp)    #whole robots's moment of inertia around axis parallel to wheel's axis n/cm2


        self.mu = 12.55    #Robot's mass with wheels kg
        self.m  = 1.65     #mass of the wheel kg
        self.M  = self.mu - 2 * self.m

        self.g  = 981  * 10**self.lin_exp     #gravity axeliration

        self.u_max = 10000 * 10**(2*self.lin_exp)
        self.time_old = 0
        self.state = np.zeros(4)
        self.control = 0
        self.old_qtt = 0
        
        # User defined parameters for saving
        self.number_of_iteration = 0
        self.all_states_ = []
        self.actions_ = []
        self.tmp_dict = {}
        
    def _compute_state_dynamics(self, time, state, action, disturb=[]):
        
        self.number_of_iteration+=1
        
#         if self.number_of_iteration%10 == 0:
#             self.number_of_iteration = 0

        self.all_states_.append(state)
        self.actions_.append(action)
            
        if time == 0.1:
            print(f'Saving at Iteration {self.number_of_iteration} in time {time}')
            pd.DataFrame(np.array(self.all_states_)).to_csv('states.csv')        
            pd.DataFrame(np.array(self.actions_)).to_csv('actions.csv')        
        
        if time == 0:
            dt = 1e-7
        else:
            dt = time - self.time_old

#         print('v', v)
#         print('dt', dt)

        def calc_I_matrix (v):
            #v - state
            I = 1.0 * np.array([[ (self.M + 2 * self.m) * self.R**2 + 2 * self.I_w,  self.M * self.R * self.L * cos (v[1])],
                                 [            self.M * self.R * self.L * cos (v[1]),       self.M * self.L**2 + self.I_psi]])
            return I


        def calc_F_matrix (v): 
            #v - state
            return np.array([- self.M * self.R * self.L * v[3]**2 * sin (v[1]),  - self.M * self.g * self.L * sin (v[1])])

        H = np.array ([2, -2])
        us = []
        u = action

        def f (v, t, us=us, u=u):
            I = calc_I_matrix(v)
            I_inv = np.linalg.inv(I)
            F = calc_F_matrix(v)
            
            if u > self.u_max:
                u = self.u_max
            if u < -self.u_max:
                u = -self.u_max

            q_tt = I_inv.dot (H * u - F)
            
            v1 = np.zeros (4)
            v1[:2] = v[2:]
            v1[2:] = q_tt  


            return v1
        
        t = np.linspace (0, dt, 3)

#         vs = odeint(f, state, t)[-1]

# #         print('vs', vs)
#         q_tt = vs[2:]
# #         print('dt', dt)
#         self._state = vs
#         q_ttt = (q_tt - self.old_qtt)/dt
#         self.old_qtt = q_tt
#         res = np.concatenate([q_tt, q_ttt], axis=None)
# #         print('res', res)

        X_t = f(state, t)
        res = X_t
        
        return res
    
    def out(self, state, time=None, action=None):
        
        delta_time = time - self.time_old
        self.integral_alpha += delta_time * state[0]
        
        return rc.array([state[0], state[1], state[2], state[3], self.integral_alpha])
        
    def reset(self):
        self.time_old = 0
        self.integral_alpha = 0
    
class SysInvertedPendulum(System):
    """
    System class: mathematical pendulum

    """

    # DEBUG ====================================
    # def __init__(self, *args, is_angle_overflow=True, **kwargs):
    # /DEBUG ===================================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "inverted-pendulum"

        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]

        self.time_old = 0
        self.integral_alpha = 0

        # DEBUG ====================================
        # self.is_angle_overflow = is_angle_overflow
        # /DEBUG ===================================

    def _compute_state_dynamics(self, time, state, action, disturb=[]):
#         print('state_compute_state_dynamics',state)

        m, g, l = self.pars[0], self.pars[1], self.pars[2]

        Dstate = rc.zeros(self.dim_state, prototype=action)
        Dstate[0] = state[1]
        Dstate[1] = g / l * rc.sin(state[0]) + action[0] / (m * l ** 2)
#         print('Dstate', Dstate)

        return Dstate

    def out(self, state, time=None, action=None):

        # DEBUG ====================================
        # observation = rc.zeros(self.dim_output)
        # observation = state[:3] + measNoise  # <-- Measure only position and orientation
        # observation = state  # <-- Position, force and torque sensors on
        # if self.is_angle_overflow:
        #     delta = np.abs(np.pi - state[0])
        #     if state[0] > 0:
        #         if state[0] > np.pi:
        #             state = [-np.pi + delta, state[1]]
        #     else:
        #         if state[0] < -np.pi:
        #             state = [np.pi - delta, state[1]]
        # /DEBUG ===================================
        delta_time = time - self.time_old
        self.integral_alpha += delta_time * state[0]

        return rc.array([state[0], self.integral_alpha, state[1]])

    def reset(self):
        self.time_old = 0
        self.integral_alpha = 0


class Sys3WRobot(System):
    """
    System class: 3-wheel robot with dynamical actuators.

    Description
    -----------
    Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

    .. math::
        \\begin{array}{ll}
                        \dot x_с & = v \cos \\angle \\newline
                        \dot y_с & = v \sin \\angle \\newline
                        \dot \\angle & = \\omega \\newline
                        \dot v & = \\left( \\frac 1 m F + q_1 \\right) \\newline
                        \dot \\omega & = \\left( \\frac 1 I M + q_2 \\right)
        \\end{array}

    **Variables**

    | :math:`x_с` : state-coordinate [m]
    | :math:`y_с` : observation-coordinate [m]
    | :math:`\\angle` : turning angle [rad]
    | :math:`v` : speed [m/s]
    | :math:`\\omega` : revolution speed [rad/s]
    | :math:`F` : pushing force [N]
    | :math:`M` : steering torque [Nm]
    | :math:`m` : robot mass [kg]
    | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
    | :math:`disturb` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``

    :math:`state = [x_c, y_c, \\angle, v, \\omega]`

    :math:`action = [F, M]`

    ``pars`` = :math:`[m, I]`

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
        nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "3wrobot"

        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        m, I = self.pars[0], self.pars[1]

        Dstate = rc.zeros(self.dim_state, prototype=action)
        Dstate[0] = state[3] * rc.cos(state[2])
        Dstate[1] = state[3] * rc.sin(state[2])
        Dstate[2] = state[4]

        if self.is_disturb and (disturb != []):
            Dstate[3] = 1 / m * (action[0] + disturb[0])
            Dstate[4] = 1 / I * (action[1] + disturb[1])
        else:
            Dstate[3] = 1 / m * action[0]
            Dstate[4] = 1 / I * action[1]

        return Dstate

    def _compute_disturbance_dynamics(self, time, disturb):

        """
        Description
        -----------

        We use here a 1st-order stochastic linear system of the type

        .. math:: \mathrm d Q_t = - \\frac{1}{\\tau_disturb} \\left( Q_t \\mathrm d t + \\sigma_disturb ( \\mathrm d B_t + \\mu_disturb ) \\right) ,

        where :math:`B` is the standard Brownian motion, :math:`Q` is the stochastic process whose realization is :math:`disturb`, and
        :math:`\\tau_disturb, \\sigma_disturb, \\mu_disturb` are the time constant, standard deviation and mean, resp.

        ``pars_disturb = [sigma_disturb, mu_disturb, tau_disturb]``, with each being an array of shape ``[dim_disturb, ]``

        """
        Ddisturb = rc.zeros(self.dim_disturb)

        for k in range(0, self.dim_disturb):
            Ddisturb[k] = -self.tau_disturb[k] * (
                disturb[k] + self.sigma_disturb[k] * (randn() + self.mu_disturb[k])
            )

        return Ddisturb

    def out(self, state, time=None, action=None):

        # observation = rc.zeros(self.dim_output)
        # observation = state[:3] + measNoise # <-- Measure only position and orientation
        # observation = state  # <-- Position, force and torque sensors on
        return state


class Sys3WRobotNI(System):
    """
    System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "3wrobotNI"

        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        Dstate = rc.zeros(self.dim_state, prototype=action)

        if self.is_disturb and (disturb != []):
            Dstate[0] = action[0] * rc.cos(state[2]) + disturb[0]
            Dstate[1] = action[0] * rc.sin(state[2]) + disturb[0]
            Dstate[2] = action[1] + disturb[1]
        else:
            Dstate[0] = action[0] * rc.cos(state[2])
            Dstate[1] = action[0] * rc.sin(state[2])
            Dstate[2] = action[1]

        return Dstate

    def _compute_disturbance_dynamics(self, time, disturb):

        """ """
        Ddisturb = rc.zeros(self.dim_disturb)

        for k in range(0, self.dim_disturb):
            Ddisturb[k] = -self.tau_disturb[k] * (
                disturb[k] + self.sigma_disturb[k] * (randn() + self.mu_disturb[k])
            )

        return Ddisturb

    def out(self, state, time=None, action=None):

        return state


class Sys2Tank(System):
    """
    Two-tank system with nonlinearity.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "2tank"

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        tau1, tau2, K1, K2, K3 = self.pars

        Dstate = rc.zeros(self.dim_state, prototype=action)
        Dstate[0] = 1 / (tau1) * (-state[0] + K1 * action)
        Dstate[1] = 1 / (tau2) * (-state[1] + K2 * state[0] + K3 * state[1] ** 2)

        return Dstate

    def _compute_disturbance_dynamics(self, time, disturb):

        Ddisturb = rc.zeros(self.dim_disturb)

        return Ddisturb

    def out(self, state, time=None, action=[]):

        return state


class GridWorld(System):
    """
    A simple 2-dimensional grid world with five actions: left, right, up, down and do nothing.
    The action encoding rule is as follows: right, left, up, down, do nothing -> 0, 1, 2, 3, 4.

    """

    def __init__(self, dims, terminal_state):
        self.dims = dims
        self.terminal_state = terminal_state

    def _compute_dynamics(self, current_state, action):
        if tuple(self.terminal_state) == tuple(current_state):
            return current_state
        if action == 0:
            if current_state[1] < self.dims[1] - 1:
                return (current_state[0], current_state[1] + 1)
        elif action == 2:
            if current_state[0] > 0:
                return (current_state[0] - 1, current_state[1])
        elif action == 1:
            if current_state[1] > 0:
                return (current_state[0], current_state[1] - 1)
        elif action == 3:
            if current_state[0] < self.dims[0] - 1:
                return (current_state[0] + 1, current_state[1])
        return current_state
