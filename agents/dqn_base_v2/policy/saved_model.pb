??	
??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
(QNetwork/EncodingNetwork/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_10/kernel
?
<QNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_10/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_10/bias
?
:QNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_10/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_11/kernel
?
<QNetwork/EncodingNetwork/dense_11/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_11/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_11/bias
?
:QNetwork/EncodingNetwork/dense_11/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_11/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_12/kernel
?
<QNetwork/EncodingNetwork/dense_12/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_12/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_12/bias
?
:QNetwork/EncodingNetwork/dense_12/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_12/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_13/kernel
?
<QNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_13/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_13/bias
?
:QNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_13/bias*
_output_shapes	
:?*
dtype0
?
QNetwork/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameQNetwork/dense_14/kernel
?
,QNetwork/dense_14/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_14/kernel*
_output_shapes
:	?*
dtype0
?
QNetwork/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameQNetwork/dense_14/bias
}
*QNetwork/dense_14/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_14/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
k
collect_data_spec

train_step
metadata
model_variables
_all_assets

signatures

observation
1
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
V
0
	1

2
3
4
5
6
7
8
9
10
11

0
1
2
 
 
QO
VARIABLE_VALUEconv2d_1/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEconv2d_1/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_10/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_10/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_11/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_11/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_12/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_12/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_13/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_13/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEQNetwork/dense_14/kernel-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEQNetwork/dense_14/bias-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

ref
1

observation
3

observation
1
;

_q_network
_time_step_spec
_trajectory_spec
?
_input_tensor_spec
_encoder
_q_value_layer
trainable_variables
 	variables
!regularization_losses
"	keras_api

observation
1
 
?
#_input_tensor_spec
$_preprocessing_nest
%_flat_preprocessing_layers
&_preprocessing_combiner
'_postprocessing_layers
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

kernel
bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
V
0
	1

2
3
4
5
6
7
8
9
10
11
V
0
	1

2
3
4
5
6
7
8
9
10
11
 
?
0layer_metrics
trainable_variables

1layers
2non_trainable_variables
3layer_regularization_losses
 	variables
4metrics
!regularization_losses
 
 

50
61
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
#
;0
<1
=2
>3
?4
F
0
	1

2
3
4
5
6
7
8
9
F
0
	1

2
3
4
5
6
7
8
9
 
?
@layer_metrics
(trainable_variables

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
)	variables
Dmetrics
*regularization_losses

0
1

0
1
 
?
Elayer_metrics
,trainable_variables

Flayers
Gnon_trainable_variables
Hlayer_regularization_losses
-	variables
Imetrics
.regularization_losses
 

0
1
 
 
 
?
Jlayer_with_weights-0
Jlayer-0
Klayer-1
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
R
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
 
 
 
?
Tlayer_metrics
7trainable_variables

Ulayers
Vnon_trainable_variables
Wlayer_regularization_losses
8	variables
Xmetrics
9regularization_losses
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
h


kernel
bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
h

kernel
bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
h

kernel
bias
etrainable_variables
f	variables
gregularization_losses
h	keras_api
h

kernel
bias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
 
8
50
61
&2
;3
<4
=5
>6
?7
 
 
 
 
 
 
 
 
h

kernel
	bias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
R
qtrainable_variables
r	variables
sregularization_losses
t	keras_api

0
	1

0
	1
 
?
ulayer_metrics
Ltrainable_variables

vlayers
wnon_trainable_variables
xlayer_regularization_losses
M	variables
ymetrics
Nregularization_losses
 
 
 
?
zlayer_metrics
Ptrainable_variables

{layers
|non_trainable_variables
}layer_regularization_losses
Q	variables
~metrics
Rregularization_losses
 
 
 
 
 
 
 
 
?
layer_metrics
Ytrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
Z	variables
?metrics
[regularization_losses


0
1


0
1
 
?
?layer_metrics
]trainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
^	variables
?metrics
_regularization_losses

0
1

0
1
 
?
?layer_metrics
atrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
b	variables
?metrics
cregularization_losses

0
1

0
1
 
?
?layer_metrics
etrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
f	variables
?metrics
gregularization_losses

0
1

0
1
 
?
?layer_metrics
itrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
j	variables
?metrics
kregularization_losses

0
	1

0
	1
 
?
?layer_metrics
mtrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
n	variables
?metrics
oregularization_losses
 
 
 
?
?layer_metrics
qtrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
r	variables
?metrics
sregularization_losses
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_0/observation/boardPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
t
action_0/observation/markPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
j
action_0/rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0/step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/boardaction_0/observation/markaction_0/rewardaction_0/step_typeconv2d_1/kernelconv2d_1/bias(QNetwork/EncodingNetwork/dense_10/kernel&QNetwork/EncodingNetwork/dense_10/bias(QNetwork/EncodingNetwork/dense_11/kernel&QNetwork/EncodingNetwork/dense_11/bias(QNetwork/EncodingNetwork/dense_12/kernel&QNetwork/EncodingNetwork/dense_12/bias(QNetwork/EncodingNetwork/dense_13/kernel&QNetwork/EncodingNetwork/dense_13/biasQNetwork/dense_14/kernelQNetwork/dense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_55958314
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_55958326
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_55958348
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_55958341
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_11/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_11/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_12/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_12/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOp,QNetwork/dense_14/kernel/Read/ReadVariableOp*QNetwork/dense_14/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_55958856
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableconv2d_1/kernelconv2d_1/bias(QNetwork/EncodingNetwork/dense_10/kernel&QNetwork/EncodingNetwork/dense_10/bias(QNetwork/EncodingNetwork/dense_11/kernel&QNetwork/EncodingNetwork/dense_11/bias(QNetwork/EncodingNetwork/dense_12/kernel&QNetwork/EncodingNetwork/dense_12/bias(QNetwork/EncodingNetwork/dense_13/kernel&QNetwork/EncodingNetwork/dense_13/biasQNetwork/dense_14/kernelQNetwork/dense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_55958905??
?
(
&__inference_signature_wrapper_55958348?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_559583442
PartitionedCall*
_input_shapes 
?
^
__inference_<lambda>_55958011
readvariableop_resource
identity	??ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
?
f
,__inference_function_with_signature_55958333
unknown
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_559580112
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
?
8
&__inference_signature_wrapper_55958326

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_559583212
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
??
?
*__inference_polymorphic_action_fn_55958252
	time_step
time_step_1
time_step_2
time_step_3
time_step_4Q
Mqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource4
0qnetwork_dense_14_matmul_readvariableop_resource5
1qnetwork_dense_14_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?(QNetwork/dense_14/BiasAdd/ReadVariableOp?'QNetwork/dense_14/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2DConv2Dtime_step_3LQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd?
5QNetwork/EncodingNetwork/sequential_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   27
5QNetwork/EncodingNetwork/sequential_1/flatten_4/Const?
7QNetwork/EncodingNetwork/sequential_1/flatten_4/ReshapeReshape?QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd:output:0>QNetwork/EncodingNetwork/sequential_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????29
7QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape?
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim?
-QNetwork/EncodingNetwork/flatten_5/ExpandDims
ExpandDimstime_step_4:QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2/
-QNetwork/EncodingNetwork/flatten_5/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_1/concat/axis?
-QNetwork/EncodingNetwork/concatenate_1/concatConcatV2@QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape:output:06QNetwork/EncodingNetwork/flatten_5/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_1/concat?
(QNetwork/EncodingNetwork/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2*
(QNetwork/EncodingNetwork/flatten_6/Const?
*QNetwork/EncodingNetwork/flatten_6/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_1/concat:output:01QNetwork/EncodingNetwork/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_6/Reshape?
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/flatten_6/Reshape:output:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_10/MatMul?
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_10/BiasAdd?
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_10/Relu?
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0?QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_11/MatMul?
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_11/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_11/MatMul:product:0@QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_11/BiasAdd?
&QNetwork/EncodingNetwork/dense_11/ReluRelu2QNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_11/Relu?
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_12/MatMulMatMul4QNetwork/EncodingNetwork/dense_11/Relu:activations:0?QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_12/MatMul?
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_12/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_12/MatMul:product:0@QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_12/BiasAdd?
&QNetwork/EncodingNetwork/dense_12/ReluRelu2QNetwork/EncodingNetwork/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_12/Relu?
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_13/MatMulMatMul4QNetwork/EncodingNetwork/dense_12/Relu:activations:0?QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_13/MatMul?
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_13/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_13/MatMul:product:0@QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_13/BiasAdd?
&QNetwork/EncodingNetwork/dense_13/ReluRelu2QNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_13/Relu?
'QNetwork/dense_14/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_14/MatMul/ReadVariableOp?
QNetwork/dense_14/MatMulMatMul4QNetwork/EncodingNetwork/dense_13/Relu:activations:0/QNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/MatMul?
(QNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_14/BiasAdd/ReadVariableOp?
QNetwork/dense_14/BiasAddBiasAdd"QNetwork/dense_14/MatMul:product:00QNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_14/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp)^QNetwork/dense_14/BiasAdd/ReadVariableOp(^QNetwork/dense_14/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp2T
(QNetwork/dense_14/BiasAdd/ReadVariableOp(QNetwork/dense_14/BiasAdd/ReadVariableOp2R
'QNetwork/dense_14/MatMul/ReadVariableOp'QNetwork/dense_14/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:ZV
/
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step
?
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_55958783

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958686
conv2d_1_input+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d_1_input&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdds
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
IdentityIdentityflatten_4/Reshape:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_1_input
?
H
,__inference_flatten_4_layer_call_fn_55958788

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_559583842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_55958777

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_559583622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_function_with_signature_55958279
	step_type

reward
discount
observation_board
observation_mark
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_boardobservation_markunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_polymorphic_action_fn_559582522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/board:WS
#
_output_shapes
:?????????
,
_user_specified_name0/observation/mark
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_55958362

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
4

__inference_<lambda>_55958014*
_input_shapes 
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958698
conv2d_1_input+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d_1_input&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdds
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
IdentityIdentityflatten_4/Reshape:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_1_input
?
8
&__inference_get_initial_state_55958674

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
/__inference_sequential_1_layer_call_fn_55958707
conv2d_1_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_559584162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_1_input
?
?
&__inference_signature_wrapper_55958314
discount
observation_board
observation_mark

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_boardobservation_markunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_559582792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:d`
/
_output_shapes
:?????????
-
_user_specified_name0/observation/board:WS
#
_output_shapes
:?????????
,
_user_specified_name0/observation/mark:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?
8
&__inference_get_initial_state_55958320

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958728

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdds
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
IdentityIdentityflatten_4/Reshape:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_55958768

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
,__inference_function_with_signature_55958344?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_559580142
PartitionedCall*
_input_shapes 
?

?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958416

inputs
conv2d_1_55958409
conv2d_1_55958411
identity?? conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_55958409conv2d_1_55958411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_559583622"
 conv2d_1/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_559583842
flatten_4/PartitionedCall?
IdentityIdentity"flatten_4/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
$__inference__traced_restore_55958905
file_prefix
assignvariableop_variable&
"assignvariableop_1_conv2d_1_kernel$
 assignvariableop_2_conv2d_1_bias?
;assignvariableop_3_qnetwork_encodingnetwork_dense_10_kernel=
9assignvariableop_4_qnetwork_encodingnetwork_dense_10_bias?
;assignvariableop_5_qnetwork_encodingnetwork_dense_11_kernel=
9assignvariableop_6_qnetwork_encodingnetwork_dense_11_bias?
;assignvariableop_7_qnetwork_encodingnetwork_dense_12_kernel=
9assignvariableop_8_qnetwork_encodingnetwork_dense_12_bias?
;assignvariableop_9_qnetwork_encodingnetwork_dense_13_kernel>
:assignvariableop_10_qnetwork_encodingnetwork_dense_13_bias0
,assignvariableop_11_qnetwork_dense_14_kernel.
*assignvariableop_12_qnetwork_dense_14_bias
identity_14??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp;assignvariableop_3_qnetwork_encodingnetwork_dense_10_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_qnetwork_encodingnetwork_dense_10_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp;assignvariableop_5_qnetwork_encodingnetwork_dense_11_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp9assignvariableop_6_qnetwork_encodingnetwork_dense_11_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_qnetwork_encodingnetwork_dense_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_qnetwork_encodingnetwork_dense_12_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp;assignvariableop_9_qnetwork_encodingnetwork_dense_13_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_qnetwork_encodingnetwork_dense_13_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_qnetwork_dense_14_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_qnetwork_dense_14_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13?
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
&__inference_signature_wrapper_55958341
unknown
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_559583332
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
??
?
*__inference_polymorphic_action_fn_55958524
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_board
time_step_observation_markQ
Mqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource4
0qnetwork_dense_14_matmul_readvariableop_resource5
1qnetwork_dense_14_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?(QNetwork/dense_14/BiasAdd/ReadVariableOp?'QNetwork/dense_14/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2DConv2Dtime_step_observation_boardLQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd?
5QNetwork/EncodingNetwork/sequential_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   27
5QNetwork/EncodingNetwork/sequential_1/flatten_4/Const?
7QNetwork/EncodingNetwork/sequential_1/flatten_4/ReshapeReshape?QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd:output:0>QNetwork/EncodingNetwork/sequential_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????29
7QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape?
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim?
-QNetwork/EncodingNetwork/flatten_5/ExpandDims
ExpandDimstime_step_observation_mark:QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2/
-QNetwork/EncodingNetwork/flatten_5/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_1/concat/axis?
-QNetwork/EncodingNetwork/concatenate_1/concatConcatV2@QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape:output:06QNetwork/EncodingNetwork/flatten_5/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_1/concat?
(QNetwork/EncodingNetwork/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2*
(QNetwork/EncodingNetwork/flatten_6/Const?
*QNetwork/EncodingNetwork/flatten_6/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_1/concat:output:01QNetwork/EncodingNetwork/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_6/Reshape?
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/flatten_6/Reshape:output:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_10/MatMul?
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_10/BiasAdd?
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_10/Relu?
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0?QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_11/MatMul?
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_11/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_11/MatMul:product:0@QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_11/BiasAdd?
&QNetwork/EncodingNetwork/dense_11/ReluRelu2QNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_11/Relu?
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_12/MatMulMatMul4QNetwork/EncodingNetwork/dense_11/Relu:activations:0?QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_12/MatMul?
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_12/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_12/MatMul:product:0@QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_12/BiasAdd?
&QNetwork/EncodingNetwork/dense_12/ReluRelu2QNetwork/EncodingNetwork/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_12/Relu?
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_13/MatMulMatMul4QNetwork/EncodingNetwork/dense_12/Relu:activations:0?QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_13/MatMul?
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_13/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_13/MatMul:product:0@QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_13/BiasAdd?
&QNetwork/EncodingNetwork/dense_13/ReluRelu2QNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_13/Relu?
'QNetwork/dense_14/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_14/MatMul/ReadVariableOp?
QNetwork/dense_14/MatMulMatMul4QNetwork/EncodingNetwork/dense_13/Relu:activations:0/QNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/MatMul?
(QNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_14/BiasAdd/ReadVariableOp?
QNetwork/dense_14/BiasAddBiasAdd"QNetwork/dense_14/MatMul:product:00QNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_14/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp)^QNetwork/dense_14/BiasAdd/ReadVariableOp(^QNetwork/dense_14/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp2T
(QNetwork/dense_14/BiasAdd/ReadVariableOp(QNetwork/dense_14/BiasAdd/ReadVariableOp2R
'QNetwork/dense_14/MatMul/ReadVariableOp'QNetwork/dense_14/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:lh
/
_output_shapes
:?????????
5
_user_specified_nametime_step/observation/board:_[
#
_output_shapes
:?????????
4
_user_specified_nametime_step/observation/mark
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958740

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdds
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeconv2d_1/BiasAdd:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
IdentityIdentityflatten_4/Reshape:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_1_layer_call_fn_55958749

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_559584162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958435

inputs
conv2d_1_55958428
conv2d_1_55958430
identity?? conv2d_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_55958428conv2d_1_55958430*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_559583622"
 conv2d_1/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_559583842
flatten_4/PartitionedCall?
IdentityIdentity"flatten_4/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
!__inference__traced_save_55958856
file_prefix'
#savev2_variable_read_readvariableop	.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_10_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_10_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_11_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_11_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_12_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_12_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_13_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_13_bias_read_readvariableop7
3savev2_qnetwork_dense_14_kernel_read_readvariableop5
1savev2_qnetwork_dense_14_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_10_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_10_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_11_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_11_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_12_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_12_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_13_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_13_bias_read_readvariableop3savev2_qnetwork_dense_14_kernel_read_readvariableop1savev2_qnetwork_dense_14_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes}
{: : :@:@:
??:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
?
/__inference_sequential_1_layer_call_fn_55958716
conv2d_1_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_559584352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_1_input
?
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_55958384

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
/__inference_sequential_1_layer_call_fn_55958758

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_559584352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
*__inference_polymorphic_action_fn_55958606
	step_type

reward
discount
observation_board
observation_markQ
Mqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource4
0qnetwork_dense_14_matmul_readvariableop_resource5
1qnetwork_dense_14_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?(QNetwork/dense_14/BiasAdd/ReadVariableOp?'QNetwork/dense_14/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2DConv2Dobservation_boardLQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd?
5QNetwork/EncodingNetwork/sequential_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   27
5QNetwork/EncodingNetwork/sequential_1/flatten_4/Const?
7QNetwork/EncodingNetwork/sequential_1/flatten_4/ReshapeReshape?QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd:output:0>QNetwork/EncodingNetwork/sequential_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????29
7QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape?
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim?
-QNetwork/EncodingNetwork/flatten_5/ExpandDims
ExpandDimsobservation_mark:QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2/
-QNetwork/EncodingNetwork/flatten_5/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_1/concat/axis?
-QNetwork/EncodingNetwork/concatenate_1/concatConcatV2@QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape:output:06QNetwork/EncodingNetwork/flatten_5/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_1/concat?
(QNetwork/EncodingNetwork/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2*
(QNetwork/EncodingNetwork/flatten_6/Const?
*QNetwork/EncodingNetwork/flatten_6/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_1/concat:output:01QNetwork/EncodingNetwork/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_6/Reshape?
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/flatten_6/Reshape:output:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_10/MatMul?
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_10/BiasAdd?
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_10/Relu?
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0?QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_11/MatMul?
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_11/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_11/MatMul:product:0@QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_11/BiasAdd?
&QNetwork/EncodingNetwork/dense_11/ReluRelu2QNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_11/Relu?
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_12/MatMulMatMul4QNetwork/EncodingNetwork/dense_11/Relu:activations:0?QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_12/MatMul?
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_12/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_12/MatMul:product:0@QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_12/BiasAdd?
&QNetwork/EncodingNetwork/dense_12/ReluRelu2QNetwork/EncodingNetwork/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_12/Relu?
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_13/MatMulMatMul4QNetwork/EncodingNetwork/dense_12/Relu:activations:0?QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_13/MatMul?
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_13/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_13/MatMul:product:0@QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_13/BiasAdd?
&QNetwork/EncodingNetwork/dense_13/ReluRelu2QNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_13/Relu?
'QNetwork/dense_14/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_14/MatMul/ReadVariableOp?
QNetwork/dense_14/MatMulMatMul4QNetwork/EncodingNetwork/dense_13/Relu:activations:0/QNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/MatMul?
(QNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_14/BiasAdd/ReadVariableOp?
QNetwork/dense_14/BiasAddBiasAdd"QNetwork/dense_14/MatMul:product:00QNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_14/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp)^QNetwork/dense_14/BiasAdd/ReadVariableOp(^QNetwork/dense_14/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp2T
(QNetwork/dense_14/BiasAdd/ReadVariableOp(QNetwork/dense_14/BiasAdd/ReadVariableOp2R
'QNetwork/dense_14/MatMul/ReadVariableOp'QNetwork/dense_14/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/board:UQ
#
_output_shapes
:?????????
*
_user_specified_nameobservation/mark
?
>
,__inference_function_with_signature_55958321

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_get_initial_state_559583202
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?q
?
0__inference_polymorphic_distribution_fn_55958671
	step_type

reward
discount
observation_board
observation_markQ
Mqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource4
0qnetwork_dense_14_matmul_readvariableop_resource5
1qnetwork_dense_14_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?(QNetwork/dense_14/BiasAdd/ReadVariableOp?'QNetwork/dense_14/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2DConv2Dobservation_boardLQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd?
5QNetwork/EncodingNetwork/sequential_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   27
5QNetwork/EncodingNetwork/sequential_1/flatten_4/Const?
7QNetwork/EncodingNetwork/sequential_1/flatten_4/ReshapeReshape?QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd:output:0>QNetwork/EncodingNetwork/sequential_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????29
7QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape?
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim?
-QNetwork/EncodingNetwork/flatten_5/ExpandDims
ExpandDimsobservation_mark:QNetwork/EncodingNetwork/flatten_5/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2/
-QNetwork/EncodingNetwork/flatten_5/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_1/concat/axis?
-QNetwork/EncodingNetwork/concatenate_1/concatConcatV2@QNetwork/EncodingNetwork/sequential_1/flatten_4/Reshape:output:06QNetwork/EncodingNetwork/flatten_5/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_1/concat?
(QNetwork/EncodingNetwork/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2*
(QNetwork/EncodingNetwork/flatten_6/Const?
*QNetwork/EncodingNetwork/flatten_6/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_1/concat:output:01QNetwork/EncodingNetwork/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_6/Reshape?
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/flatten_6/Reshape:output:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_10/MatMul?
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_10/BiasAdd?
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_10/Relu?
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0?QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_11/MatMul?
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_11/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_11/MatMul:product:0@QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_11/BiasAdd?
&QNetwork/EncodingNetwork/dense_11/ReluRelu2QNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_11/Relu?
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_12/MatMulMatMul4QNetwork/EncodingNetwork/dense_11/Relu:activations:0?QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_12/MatMul?
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_12/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_12/MatMul:product:0@QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_12/BiasAdd?
&QNetwork/EncodingNetwork/dense_12/ReluRelu2QNetwork/EncodingNetwork/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_12/Relu?
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_13/MatMulMatMul4QNetwork/EncodingNetwork/dense_12/Relu:activations:0?QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_13/MatMul?
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_13/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_13/MatMul:product:0@QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_13/BiasAdd?
&QNetwork/EncodingNetwork/dense_13/ReluRelu2QNetwork/EncodingNetwork/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_13/Relu?
'QNetwork/dense_14/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_14/MatMul/ReadVariableOp?
QNetwork/dense_14/MatMulMatMul4QNetwork/EncodingNetwork/dense_13/Relu:activations:0/QNetwork/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/MatMul?
(QNetwork/dense_14/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_14/BiasAdd/ReadVariableOp?
QNetwork/dense_14/BiasAddBiasAdd"QNetwork/dense_14/MatMul:product:00QNetwork/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_14/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_14/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMax?
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtol?
IdentityIdentityCategorical_1/mode/Cast:y:09^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp)^QNetwork/dense_14/BiasAdd/ReadVariableOp(^QNetwork/dense_14/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_12/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_12/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_13/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_13/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_1/conv2d_1/Conv2D/ReadVariableOp2T
(QNetwork/dense_14/BiasAdd/ReadVariableOp(QNetwork/dense_14/BiasAdd/ReadVariableOp2R
'QNetwork/dense_14/MatMul/ReadVariableOp'QNetwork/dense_14/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:b^
/
_output_shapes
:?????????
+
_user_specified_nameobservation/board:UQ
#
_output_shapes
:?????????
*
_user_specified_nameobservation/mark"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0/discount:0?????????
R
0/observation/board;
action_0/observation/board:0?????????
D
0/observation/mark.
action_0/observation/mark:0?????????
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????6
action,
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
?
collect_data_spec

train_step
metadata
model_variables
_all_assets

signatures
?action
?distribution
?get_initial_state
?get_metadata
?get_train_step"
_generic_user_object
9
observation
1"
trackable_tuple_wrapper
:	 (2Variable
 "
trackable_dict_wrapper
w
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_tuple_wrapper
5
0
1
2"
trackable_list_wrapper
d
?action
?get_initial_state
?get_train_step
?get_metadata"
signature_map
 "
trackable_dict_wrapper
):'@2conv2d_1/kernel
:@2conv2d_1/bias
<::
??2(QNetwork/EncodingNetwork/dense_10/kernel
5:3?2&QNetwork/EncodingNetwork/dense_10/bias
<::
??2(QNetwork/EncodingNetwork/dense_11/kernel
5:3?2&QNetwork/EncodingNetwork/dense_11/bias
<::
??2(QNetwork/EncodingNetwork/dense_12/kernel
5:3?2&QNetwork/EncodingNetwork/dense_12/bias
<::
??2(QNetwork/EncodingNetwork/dense_13/kernel
5:3?2&QNetwork/EncodingNetwork/dense_13/bias
+:)	?2QNetwork/dense_14/kernel
$:"2QNetwork/dense_14/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
9
observation
3"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper
Y

_q_network
_time_step_spec
_trajectory_spec"
_generic_user_object
?
_input_tensor_spec
_encoder
_q_value_layer
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
9
observation
1"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
?
#_input_tensor_spec
$_preprocessing_nest
%_flat_preprocessing_layers
&_preprocessing_combiner
'_postprocessing_layers
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

kernel
bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
v
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
	1

2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0layer_metrics
trainable_variables

1layers
2non_trainable_variables
3layer_regularization_losses
 	variables
4metrics
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 768]}, {"class_name": "TensorShape", "items": [1, 1]}]}
C
;0
<1
=2
>3
?4"
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@layer_metrics
(trainable_variables

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
)	variables
Dmetrics
*regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Elayer_metrics
,trainable_variables

Flayers
Gnon_trainable_variables
Hlayer_regularization_losses
-	variables
Imetrics
.regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jlayer_with_weights-0
Jlayer-0
Klayer-1
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 7, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}}}
?
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tlayer_metrics
7trainable_variables

Ulayers
Vnon_trainable_variables
Wlayer_regularization_losses
8	variables
Xmetrics
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?


kernel
bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 769}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 769]}}
?

kernel
bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
?

kernel
bias
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
?

kernel
bias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
 "
trackable_dict_wrapper
X
50
61
&2
;3
<4
=5
>6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?	

kernel
	bias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 7, 1]}}
?
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ulayer_metrics
Ltrainable_variables

vlayers
wnon_trainable_variables
xlayer_regularization_losses
M	variables
ymetrics
Nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
zlayer_metrics
Ptrainable_variables

{layers
|non_trainable_variables
}layer_regularization_losses
Q	variables
~metrics
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
Ytrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
Z	variables
?metrics
[regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
]trainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
^	variables
?metrics
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
atrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
b	variables
?metrics
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
etrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
f	variables
?metrics
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
itrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
j	variables
?metrics
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
mtrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
n	variables
?metrics
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
qtrainable_variables
?layers
?non_trainable_variables
 ?layer_regularization_losses
r	variables
?metrics
sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
*__inference_polymorphic_action_fn_55958524
*__inference_polymorphic_action_fn_55958606?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_polymorphic_distribution_fn_55958671?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_get_initial_state_55958674?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_55958014"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_55958011"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_55958314
0/discount0/observation/board0/observation/mark0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_55958326
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_55958341"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_55958348"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958728
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958740
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958686
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958698?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_1_layer_call_fn_55958758
/__inference_sequential_1_layer_call_fn_55958749
/__inference_sequential_1_layer_call_fn_55958716
/__inference_sequential_1_layer_call_fn_55958707?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_1_layer_call_fn_55958777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_55958768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_4_layer_call_fn_55958788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_4_layer_call_and_return_conditional_losses_55958783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 <
__inference_<lambda>_55958011?

? 
? "? 	5
__inference_<lambda>_55958014?

? 
? "? ?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_55958768l	7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_1_layer_call_fn_55958777_	7?4
-?*
(?%
inputs?????????
? " ??????????@?
G__inference_flatten_4_layer_call_and_return_conditional_losses_55958783a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
,__inference_flatten_4_layer_call_fn_55958788T7?4
-?*
(?%
inputs?????????@
? "???????????S
&__inference_get_initial_state_55958674)"?
?
?

batch_size 
? "? ?
*__inference_polymorphic_action_fn_55958524?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount??????????
observation???
F
board=?:
time_step/observation/board?????????
8
mark0?-
time_step/observation/mark?????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
*__inference_polymorphic_action_fn_55958606?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observationq?n
<
board3?0
observation/board?????????
.
mark&?#
observation/mark?????????
? 
? "R?O

PolicyStep&
action?
action?????????
state? 
info? ?
0__inference_polymorphic_distribution_fn_55958671?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount??????????
observationq?n
<
board3?0
observation/board?????????
.
mark&?#
observation/mark?????????
? 
? "???

PolicyStep?
action?????Ã}?z
`
C?@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*?'
%
loc?
Identity?????????
? _TFPTypeSpec
state? 
info? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958686u	G?D
=?:
0?-
conv2d_1_input?????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958698u	G?D
=?:
0?-
conv2d_1_input?????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958728m	??<
5?2
(?%
inputs?????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_55958740m	??<
5?2
(?%
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
/__inference_sequential_1_layer_call_fn_55958707h	G?D
=?:
0?-
conv2d_1_input?????????
p

 
? "????????????
/__inference_sequential_1_layer_call_fn_55958716h	G?D
=?:
0?-
conv2d_1_input?????????
p 

 
? "????????????
/__inference_sequential_1_layer_call_fn_55958749`	??<
5?2
(?%
inputs?????????
p

 
? "????????????
/__inference_sequential_1_layer_call_fn_55958758`	??<
5?2
(?%
inputs?????????
p 

 
? "????????????
&__inference_signature_wrapper_55958314?	
???
? 
???
.

0/discount ?

0/discount?????????
L
0/observation/board5?2
0/observation/board?????????
>
0/observation/mark(?%
0/observation/mark?????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"+?(
&
action?
action?????????a
&__inference_signature_wrapper_5595832670?-
? 
&?#
!

batch_size?

batch_size "? Z
&__inference_signature_wrapper_559583410?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_55958348?

? 
? "? 