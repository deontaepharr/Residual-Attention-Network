from keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization

from keras.models import Model

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
            padded_input_data = ZeroPadding2D( (x_dim_inc, y_dim_inc) )(input_data)
            conv_layer_1 = self.convolution_layer(conv_input_data=padded_input_data, 
                                filters=64, 
                                kernel_size=(1,1), 
                                strides=(1,1), 
                                padding='same')
        else:
            conv_layer_1 = self.convolution_layer(conv_input_data=input_data, 
                                filters=64, 
                                kernel_size=(1,1), 
                                strides=(1,1), 
                                padding='same')
        
        max_pool_layer_1 = self.max_pool_layer(conv_layer_1)

        # Residual Unit then Attention Module #1
        res_unit_1 = self.residual_unit(max_pool_layer_1, filters=[32, 32, 64])
        att_mod_1 = self.attention_module(res_unit_1, filters=[32, 32, 64])
        
        # Residual Unit then Attention Module #2
        res_unit_2 = self.residual_unit(att_mod_1, filters=[32, 32, 64])
        att_mod_2 = self.attention_module(res_unit_2, filters=[32, 32, 64])

        # Residual Unit then Attention Module #3
        res_unit_3 = self.residual_unit(att_mod_2, filters=[32, 32, 64])
        att_mod_3 = self.attention_module(res_unit_3, filters=[32, 32, 64])

        # Ending it all
        res_unit_end_1 = self.residual_unit(att_mod_3, filters=[32, 32, 64])
        res_unit_end_2 = self.residual_unit(res_unit_end_1, filters=[32, 32, 64])
        res_unit_end_3 = self.residual_unit(res_unit_end_2, filters=[32, 32, 64])
        res_unit_end_4 = self.residual_unit(res_unit_end_3, filters=[32, 32, 64])

        # Avg Pooling
        avg_pool_layer = self.avg_pool_layer(res_unit_end_4)

        # Flatten the data
        flatten_op = Flatten()(avg_pool_layer)

        # FC Layer for prediction
        fully_connected_layers = Dense(self.n_classes, activation='softmax')(flatten_op)
         
        # Fully constructed model
        model = Model(inputs=input_data, outputs=fully_connected_layers)
        
        return model

    # Identity ResUnit with Bottle Neck 
    def residual_unit(self, residual_input_data, filters):
        # Hold input_x here for later processing
        residual_x = residual_input_data
        
        filter1,filter2,filter3 = filters

        # Layer 1
        res_conv_1 = self.convolution_layer(
                            conv_input_data=residual_input_data, 
                            filters=filter1, 
                            kernel_size=(1,1), 
                            strides=(1,1), 
                            padding='same')

        # Layer 2
        res_conv_2 = self.convolution_layer(conv_input_data=res_conv_1, 
                                            filters=filter2, 
                                            kernel_size=(3,3), 
                                            strides=(1,1),
                                            padding='same')

        # Layer 3: Connecting Layer
        output = self.convolution_layer(
                        conv_input_data=res_conv_2, 
                        filters=filter3, 
                        kernel_size=(1,1), 
                        strides=(1,1), 
                        padding='same', 
                        residual_x=residual_x)

        return output

    def attention_module(self, attention_input_data, filters):
        # Send input_x through 
        #p residual_units
        p_res_unit_op_1 = attention_input_data
        for i in range(self.p):
            p_res_unit_op_1 = self.residual_unit(p_res_unit_op_1, filters=filters)

        # Perform Trunk Branch Operation
        trunk_branch_op = self.trunk_branch(trunk_input_data=p_res_unit_op_1, filters=filters)

        # Perform Mask Branch Operation
        mask_branch_op = self.mask_branch(mask_input_data=p_res_unit_op_1, filters=filters)

        # Perform Attention Residual Learning: Combine Trunk and Mask branch results
        ar_learning_op = self.attention_residual_learning(mask_input=mask_branch_op, trunk_input=trunk_branch_op)

        # Send branch results through #p residual_units
        p_res_unit_op_2 = ar_learning_op
        for _ in range(self.p):
            p_res_unit_op_2 = self.residual_unit(p_res_unit_op_2, filters=filters)

        return p_res_unit_op_2

    def trunk_branch(self, trunk_input_data, filters):
        # sequence of residual units
        t_res_unit_op = trunk_input_data
        for _ in range(self.t):
            t_res_unit_op = self.residual_unit(t_res_unit_op, filters=filters)

        return t_res_unit_op

    def mask_branch(self, mask_input_data, filters, m=3):
        # r = num of residual units between adjacent pooling layers
        # m = num max pooling / linear interpolations to do

        # Downsampling Step Initialization - Top
        downsampling = self.max_pool_layer(pool_input_data=mask_input_data)

        # Perform residual units ops r times between adjacent pooling layers
        for j in range(self.r):
            downsampling = self.residual_unit(residual_input_data=downsampling, filters=filters)

        # Last pooling step before middle step - Bottom
        downsampling = self.max_pool_layer(pool_input_data=downsampling)

        # Middle Residuals - Perform 2*r residual units steps before upsampling
        middleware = downsampling
        for _ in range(2 * self.r):
            middleware = self.residual_unit(residual_input_data=middleware, filters=filters)

        # Upsampling Step Initialization - Top
        upsampling = self.upsampling_layer(upsampling_input_data=middleware)

        # Perform residual units ops r times between adjacent pooling layers
        for j in range(self.r):
            upsampling = self.residual_unit(residual_input_data=upsampling, filters=filters)

        # Last interpolation step - Bottom
        upsampling = self.upsampling_layer(upsampling_input_data=upsampling)
        conv_filter = upsampling.shape[-1].value

        conv1 = self.convolution_layer(conv_input_data=upsampling,          
                                       filters=conv_filter,
                                       kernel_size=(1,1), 
                                       strides=(1,1), 
                                       padding='same')
        
        conv2 = self.convolution_layer(conv_input_data=conv1,                                        
                                       filters=conv_filter,
                                       kernel_size=(1,1), 
                                       strides=(1,1), 
                                       padding='same')

        sigmoid = Activation('sigmoid')(conv2)

        return sigmoid

    def attention_residual_learning(self, mask_input, trunk_input):
        # https://stackoverflow.com/a/53361303/9221241
        Mx = Lambda(lambda x: 1 + x)(mask_input) # 1 + mask
        return Multiply()([Mx, trunk_input]) # M(x) * T(x)
    
    def convolution_layer(self, conv_input_data, filters, kernel_size, strides, padding, residual_x=None):

        conv_op = Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding)(conv_input_data)

        batch_op = BatchNormalization()(conv_op)
        
        if residual_x is not None:
            # Combine processed_x with residual_x
            conv_residual_x = Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding)(residual_x)

            add_op = Add()([batch_op, conv_residual_x])
            activation_op = Activation('relu')(add_op)
            
        else:
            activation_op = Activation('relu')(batch_op)

        return activation_op

    def max_pool_layer(self, pool_input_data, pool_size=(2, 2), strides=(2, 2)):
        return MaxPool2D(pool_size=pool_size,
                         strides=strides,
                         padding='same')(pool_input_data)

    def avg_pool_layer(self, pool_input_data, pool_size=(2, 2), strides=(2, 2)):
        return AveragePooling2D(pool_size=pool_size, strides=strides)(pool_input_data)
    
    def upsampling_layer(self, upsampling_input_data, size=(2, 2)):
        return UpSampling2D(size=size)(upsampling_input_data)