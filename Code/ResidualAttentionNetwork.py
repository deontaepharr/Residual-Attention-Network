from tensorflow.keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization

from tensorflow.keras.models import Model

# Todo: Make scalable/all-encompassing
class ResidualAttentionNetwork():

    def __init__(self, input_shape, n_classes, p=1, t=2, r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.p = p
        self.t = t
        self.r = r
        
        
    def build_model(self):
         # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)
        
        # Initial Layers before Attention Module
        
        # Doing padding because I'm having trouble with img dims that are <= 28
        if self.input_shape[0] <= 28 or self.input_shape[1] <= 28:
            x_dim_inc = (32 - self.input_shape[0]) // 2
            y_dim_inc = (32 - self.input_shape[1]) // 2
            padded_input_data = ZeroPadding2D( (x_dim_inc,y_dim_inc) )(input_data)
            conv_layer_1 = self.convolution_layer(conv_input_data=padded_input_data)
        else:
            conv_layer_1 = self.convolution_layer(conv_input_data=input_data)
            
        
        
        max_pool_layer_1 = self.max_pool_layer(conv_layer_1)

        # Residual Unit then Attention Module #1
        res_unit_1 = self.residual_unit(max_pool_layer_1)
        
        att_mod_1 = self.attention_module(res_unit_1, self.p, self.t, self.r)
        
        # Residual Unit then Attention Module #2
        res_unit_2 = self.residual_unit(att_mod_1)
        att_mod_2 = self.attention_module(res_unit_2, self.p, self.t, self.r)

        # Residual Unit then Attention Module #3
        res_unit_3 = self.residual_unit(att_mod_2)
        att_mod_3 = self.attention_module(res_unit_3, self.p, self.t, self.r)

        # Ending it all
        res_unit_end_1 = self.residual_unit(att_mod_3)
        res_unit_end_2 = self.residual_unit(res_unit_end_1)
        res_unit_end_3 = self.residual_unit(res_unit_end_2)
        res_unit_end_4 = self.residual_unit(res_unit_end_3)

        # Avg Pooling
        avg_pool_layer = self.avg_pool_layer(res_unit_end_4)

        # Flatten the data
        flatten_op = Flatten()(avg_pool_layer)

        # FC Layer for prediction
        fully_connected_layers = Dense(self.n_classes, activation='softmax')(flatten_op)

        # Fully constructed model
        model = Model(inputs=input_data, outputs=fully_connected_layers)
        
        return model
        
        
    def convolution_layer(self, conv_input_data, filters=32, kernel_size=(3, 3), strides=(1, 1)):

        conv_op = Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same')(conv_input_data)

        batch_op = BatchNormalization()(conv_op)

        activation_op = Activation('relu')(batch_op)

        return activation_op

    def max_pool_layer(self, pool_input_data, pool_size=(2, 2), strides=(2, 2)):
        return MaxPool2D(pool_size=pool_size,
                         strides=strides,
                         padding='same')(pool_input_data)

    def avg_pool_layer(self, pool_input_data, pool_size=(2, 2), strides=(2, 2)):
        return AveragePooling2D(pool_size=pool_size,
                                strides=strides,
                                padding='same')(pool_input_data)

    def upsampling_layer(self, upsampling_input_data, size=(2, 2), interpolation='bilinear'):
        return UpSampling2D(size=size,
                            interpolation=interpolation)(upsampling_input_data)

    # Identity ResUnit 
    def residual_unit(self, residual_input_data):
        # Hold input_x here for later processing
        skipped_x = residual_input_data

        # Layer 1
        res_conv_1 = self.convolution_layer(conv_input_data=residual_input_data, filters=32)

        # Layer 2
        res_conv_2 = self.convolution_layer(conv_input_data=res_conv_1, filters=64)

        # Connecting Layer
        output = self.connecting_residual_layer(conn_input_data=res_conv_2, skipped_x=skipped_x)

        return output

    def connecting_residual_layer(self, conn_input_data, skipped_x, filters=32, kernel_size=(5, 5), strides=(1, 1)):
        # Connecting Layer
        conv_op = Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same')(conn_input_data)

        batch_op = BatchNormalization()(conv_op)
        
        # Combine processed_x with skipped_x
        add_op = Add()([batch_op, skipped_x])

        activation_op = Activation('relu')(add_op)

        return activation_op

    def attention_module(self, attention_input_data, p, t, r):

        # Send input_x through #p residual_units
        p_res_unit_op_1 = attention_input_data
        for i in range(p):
            p_res_unit_op_1 = self.residual_unit(p_res_unit_op_1)

        # Perform Trunk Branch Operation
        trunk_branch_op = self.trunk_branch(trunk_input_data=p_res_unit_op_1, t=t)

        # Perform Mask Branch Operation
        mask_branch_op = self.mask_branch(mask_input_data=p_res_unit_op_1, r=r)

        # Perform Attention Residual Learning: Combine Trunk and Mask branch results
        ar_learning_op = self.attention_residual_learning(mask_input=mask_branch_op, trunk_input=trunk_branch_op)

        # Send branch results through #p residual_units
        p_res_unit_op_2 = ar_learning_op
        for _ in range(p):
            p_res_unit_op_2 = self.residual_unit(p_res_unit_op_2)

        return p_res_unit_op_2

    def trunk_branch(self, trunk_input_data, t):
        # sequence of residual units
        t_res_unit_op = trunk_input_data
        for _ in range(t):
            t_res_unit_op = self.residual_unit(t_res_unit_op)

        return t_res_unit_op

    def mask_branch(self, mask_input_data, r, m=3):
        # r = num of residual units between adjacent pooling layers
        # m = num max pooling / linear interpolations to do

        # Downsampling Step Initialization - Top
        downsampling = self.max_pool_layer(pool_input_data=mask_input_data)

        # Perform residual units ops r times between adjacent pooling layers
        for j in range(r):
            downsampling = self.residual_unit(residual_input_data=downsampling)

        # Last pooling step before middle step - Bottom
        downsampling = self.max_pool_layer(pool_input_data=downsampling)

        # Middle Residuals - Perform 2*r residual units steps before upsampling
        middleware = downsampling
        for _ in range(2 * r):
            middleware = self.residual_unit(residual_input_data=middleware)

        # Upsampling Step Initialization - Top
        upsampling = self.upsampling_layer(upsampling_input_data=middleware)

        # Perform residual units ops r times between adjacent pooling layers
        for j in range(r):
            upsampling = self.residual_unit(residual_input_data=upsampling)

        # Last interpolation step - Bottom
        upsampling = self.upsampling_layer(upsampling_input_data=upsampling)

        conv1 = self.convolution_layer(conv_input_data=upsampling, kernel_size=(1, 1))
        conv2 = self.convolution_layer(conv_input_data=conv1, kernel_size=(1, 1))

        sigmoid = Activation('sigmoid')(conv2)

        return sigmoid

    def attention_residual_learning(self, mask_input, trunk_input):
        # https://stackoverflow.com/a/53361303/9221241
        m = Lambda(lambda x: 1 + x)(mask_input) # 1 + mask
        return Multiply()([m, trunk_input]) # M(x) * T(x)