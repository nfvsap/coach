
import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.q_head import QHead
from rl_coach.base_parameters import AgentParameters
from rl_coach.spaces import SpacesDefinition


class DuelingQHeadCustom(QHead):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, number_layers: int = 1, units: int = 1):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'dueling_q_head_custom'
        self.number_layers = number_layers
        self.units = units

    def build_module(self, input_layer):
        # state value tower - V
        with tf.variable_scope("state_value"):
            self.state_value = self.dense_layer(self.units)(input_layer, activation=self.activation_function, name='fc1') _
            for i in range(1,self.number_layers):
                self.state_value = self.dense_layer(self.units)(input_layer, activation=self.activation_function,
                                                            name='fc1_{}'.format(str(i)))
            self.state_value = self.dense_layer(1)(self.state_value, name='fc2')

        # action advantage tower - A
        with tf.variable_scope("action_advantage"):
            self.action_advantage = self.dense_layer(self.units)(input_layer, activation=self.activation_function, name='fc1')
            for i in range(1,self.number_layers):
                self.action_advantage = self.dense_layer(self.units)(input_layer, activation=self.activation_function, name='fc1_{}'.format(str(i)))
            self.action_advantage = self.dense_layer(self.num_actions)(self.action_advantage, name='fc2')
            self.action_mean = tf.reduce_mean(self.action_advantage, axis=1, keepdims=True)
            self.action_advantage = self.action_advantage - self.action_mean

        # merge to state-action value function Q
        self.q_values = self.output = tf.add(self.state_value, self.action_advantage, name='output')

        # used in batch-rl to estimate a probablity distribution over actions
        self.softmax = self.add_softmax_with_temperature()

    def __str__(self):
        result = [
            "State Value Stream - V",
            "\tDense (num outputs = "+str(self.units)+")",
            "\tDense (num outputs = 1)",
            "Action Advantage Stream - A",
            "\tDense (num outputs = "+str(self.units)+")",
            "\tDense (num outputs = {})".format(self.num_actions),
            "\tSubtract(A, Mean(A))".format(self.num_actions),
            "Add (V, A)"
        ]
        return '\n'.join(result)

