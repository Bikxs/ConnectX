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
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
(QNetwork/EncodingNetwork/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_30/kernel
?
<QNetwork/EncodingNetwork/dense_30/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_30/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_30/bias
?
:QNetwork/EncodingNetwork/dense_30/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_30/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_31/kernel
?
<QNetwork/EncodingNetwork/dense_31/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_31/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_31/bias
?
:QNetwork/EncodingNetwork/dense_31/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_31/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_32/kernel
?
<QNetwork/EncodingNetwork/dense_32/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_32/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_32/bias
?
:QNetwork/EncodingNetwork/dense_32/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_32/bias*
_output_shapes	
:?*
dtype0
?
(QNetwork/EncodingNetwork/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_33/kernel
?
<QNetwork/EncodingNetwork/dense_33/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_33/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_33/bias
?
:QNetwork/EncodingNetwork/dense_33/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_33/bias*
_output_shapes	
:?*
dtype0
?
QNetwork/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameQNetwork/dense_34/kernel
?
,QNetwork/dense_34/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_34/kernel*
_output_shapes
:	?*
dtype0
?
QNetwork/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameQNetwork/dense_34/bias
}
*QNetwork/dense_34/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_34/bias*
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
VARIABLE_VALUEconv2d_3/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEconv2d_3/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_30/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_30/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_31/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_31/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_32/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_32/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_33/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_33/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEQNetwork/dense_34/kernel-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEQNetwork/dense_34/bias-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/boardaction_0/observation/markaction_0/rewardaction_0/step_typeconv2d_3/kernelconv2d_3/bias(QNetwork/EncodingNetwork/dense_30/kernel&QNetwork/EncodingNetwork/dense_30/bias(QNetwork/EncodingNetwork/dense_31/kernel&QNetwork/EncodingNetwork/dense_31/bias(QNetwork/EncodingNetwork/dense_32/kernel&QNetwork/EncodingNetwork/dense_32/bias(QNetwork/EncodingNetwork/dense_33/kernel&QNetwork/EncodingNetwork/dense_33/biasQNetwork/dense_34/kernelQNetwork/dense_34/bias*
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
GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_167935080
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
GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_167935092
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
GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_167935114
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
GPU2*0J 8? *0
f+R)
'__inference_signature_wrapper_167935107
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_30/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_30/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_31/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_31/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_32/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_32/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_33/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_33/bias/Read/ReadVariableOp,QNetwork/dense_34/kernel/Read/ReadVariableOp*QNetwork/dense_34/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_save_167935622
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableconv2d_3/kernelconv2d_3/bias(QNetwork/EncodingNetwork/dense_30/kernel&QNetwork/EncodingNetwork/dense_30/bias(QNetwork/EncodingNetwork/dense_31/kernel&QNetwork/EncodingNetwork/dense_31/bias(QNetwork/EncodingNetwork/dense_32/kernel&QNetwork/EncodingNetwork/dense_32/bias(QNetwork/EncodingNetwork/dense_33/kernel&QNetwork/EncodingNetwork/dense_33/biasQNetwork/dense_34/kernelQNetwork/dense_34/bias*
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
GPU2*0J 8? *.
f)R'
%__inference__traced_restore_167935671??
?(
?
"__inference__traced_save_167935622
file_prefix'
#savev2_variable_read_readvariableop	.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_30_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_30_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_31_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_31_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_32_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_32_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_33_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_33_bias_read_readvariableop7
3savev2_qnetwork_dense_34_kernel_read_readvariableop5
1savev2_qnetwork_dense_34_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_30_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_30_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_31_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_31_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_32_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_32_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_33_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_33_bias_read_readvariableop3savev2_qnetwork_dense_34_kernel_read_readvariableop1savev2_qnetwork_dense_34_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
_
__inference_<lambda>_167934777
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
?;
?
%__inference__traced_restore_167935671
file_prefix
assignvariableop_variable&
"assignvariableop_1_conv2d_3_kernel$
 assignvariableop_2_conv2d_3_bias?
;assignvariableop_3_qnetwork_encodingnetwork_dense_30_kernel=
9assignvariableop_4_qnetwork_encodingnetwork_dense_30_bias?
;assignvariableop_5_qnetwork_encodingnetwork_dense_31_kernel=
9assignvariableop_6_qnetwork_encodingnetwork_dense_31_bias?
;assignvariableop_7_qnetwork_encodingnetwork_dense_32_kernel=
9assignvariableop_8_qnetwork_encodingnetwork_dense_32_bias?
;assignvariableop_9_qnetwork_encodingnetwork_dense_33_kernel>
:assignvariableop_10_qnetwork_encodingnetwork_dense_33_bias0
,assignvariableop_11_qnetwork_dense_34_kernel.
*assignvariableop_12_qnetwork_dense_34_bias
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
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_3_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_3_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp;assignvariableop_3_qnetwork_encodingnetwork_dense_30_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_qnetwork_encodingnetwork_dense_30_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp;assignvariableop_5_qnetwork_encodingnetwork_dense_31_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp9assignvariableop_6_qnetwork_encodingnetwork_dense_31_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_qnetwork_encodingnetwork_dense_32_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_qnetwork_encodingnetwork_dense_32_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp;assignvariableop_9_qnetwork_encodingnetwork_dense_33_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_qnetwork_encodingnetwork_dense_33_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_qnetwork_dense_34_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_qnetwork_dense_34_biasIdentity_12:output:0"/device:CPU:0*
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
??
?
+__inference_polymorphic_action_fn_167935018
	time_step
time_step_1
time_step_2
time_step_3
time_step_4Q
Mqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource4
0qnetwork_dense_34_matmul_readvariableop_resource5
1qnetwork_dense_34_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?(QNetwork/dense_34/BiasAdd/ReadVariableOp?'QNetwork/dense_34/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2DConv2Dtime_step_3LQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd?
6QNetwork/EncodingNetwork/sequential_3/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6QNetwork/EncodingNetwork/sequential_3/flatten_12/Const?
8QNetwork/EncodingNetwork/sequential_3/flatten_12/ReshapeReshape?QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd:output:0?QNetwork/EncodingNetwork/sequential_3/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2:
8QNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape?
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim?
.QNetwork/EncodingNetwork/flatten_13/ExpandDims
ExpandDimstime_step_4;QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????20
.QNetwork/EncodingNetwork/flatten_13/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_3/concat/axis?
-QNetwork/EncodingNetwork/concatenate_3/concatConcatV2AQNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape:output:07QNetwork/EncodingNetwork/flatten_13/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_3/concat?
)QNetwork/EncodingNetwork/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2+
)QNetwork/EncodingNetwork/flatten_14/Const?
+QNetwork/EncodingNetwork/flatten_14/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_3/concat:output:02QNetwork/EncodingNetwork/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2-
+QNetwork/EncodingNetwork/flatten_14/Reshape?
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_30/MatMulMatMul4QNetwork/EncodingNetwork/flatten_14/Reshape:output:0?QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_30/MatMul?
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_30/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_30/MatMul:product:0@QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_30/BiasAdd?
&QNetwork/EncodingNetwork/dense_30/ReluRelu2QNetwork/EncodingNetwork/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_30/Relu?
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_31/MatMulMatMul4QNetwork/EncodingNetwork/dense_30/Relu:activations:0?QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_31/MatMul?
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_31/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_31/MatMul:product:0@QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_31/BiasAdd?
&QNetwork/EncodingNetwork/dense_31/ReluRelu2QNetwork/EncodingNetwork/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_31/Relu?
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_32/MatMulMatMul4QNetwork/EncodingNetwork/dense_31/Relu:activations:0?QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_32/MatMul?
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_32/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_32/MatMul:product:0@QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_32/BiasAdd?
&QNetwork/EncodingNetwork/dense_32/ReluRelu2QNetwork/EncodingNetwork/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_32/Relu?
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_33/MatMulMatMul4QNetwork/EncodingNetwork/dense_32/Relu:activations:0?QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_33/MatMul?
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_33/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_33/MatMul:product:0@QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_33/BiasAdd?
&QNetwork/EncodingNetwork/dense_33/ReluRelu2QNetwork/EncodingNetwork/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_33/Relu?
'QNetwork/dense_34/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_34_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_34/MatMul/ReadVariableOp?
QNetwork/dense_34/MatMulMatMul4QNetwork/EncodingNetwork/dense_33/Relu:activations:0/QNetwork/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/MatMul?
(QNetwork/dense_34/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_34/BiasAdd/ReadVariableOp?
QNetwork/dense_34/BiasAddBiasAdd"QNetwork/dense_34/MatMul:product:00QNetwork/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_34/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
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
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp)^QNetwork/dense_34/BiasAdd/ReadVariableOp(^QNetwork/dense_34/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp2T
(QNetwork/dense_34/BiasAdd/ReadVariableOp(QNetwork/dense_34/BiasAdd/ReadVariableOp2R
'QNetwork/dense_34/MatMul/ReadVariableOp'QNetwork/dense_34/MatMul/ReadVariableOp:N J
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
?
a
'__inference_signature_wrapper_167935107
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
GPU2*0J 8? *6
f1R/
-__inference_function_with_signature_1679350992
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
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935494
conv2d_3_input+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_3_input&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshapeconv2d_3/BiasAdd:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
IdentityIdentityflatten_12/Reshape:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_3_input
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935506
conv2d_3_input+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_3_input&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshapeconv2d_3/BiasAdd:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
IdentityIdentityflatten_12/Reshape:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:?????????
(
_user_specified_nameconv2d_3_input
?
?
-__inference_function_with_signature_167935087

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
GPU2*0J 8? *0
f+R)
'__inference_get_initial_state_1679350862
PartitionedCall*
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
0__inference_sequential_3_layer_call_fn_167935482

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
GPU2*0J 8? *T
fORM
K__inference_sequential_3_layer_call_and_return_conditional_losses_1679352012
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
G__inference_conv2d_3_layer_call_and_return_conditional_losses_167935534

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
?
e
I__inference_flatten_12_layer_call_and_return_conditional_losses_167935150

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
?
g
-__inference_function_with_signature_167935099
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
GPU2*0J 8? *'
f"R 
__inference_<lambda>_1679347772
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
+__inference_polymorphic_action_fn_167935290
	step_type

reward
discount
observation_board
observation_markQ
Mqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource4
0qnetwork_dense_34_matmul_readvariableop_resource5
1qnetwork_dense_34_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?(QNetwork/dense_34/BiasAdd/ReadVariableOp?'QNetwork/dense_34/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2DConv2Dobservation_boardLQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd?
6QNetwork/EncodingNetwork/sequential_3/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6QNetwork/EncodingNetwork/sequential_3/flatten_12/Const?
8QNetwork/EncodingNetwork/sequential_3/flatten_12/ReshapeReshape?QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd:output:0?QNetwork/EncodingNetwork/sequential_3/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2:
8QNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape?
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim?
.QNetwork/EncodingNetwork/flatten_13/ExpandDims
ExpandDimsobservation_mark;QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????20
.QNetwork/EncodingNetwork/flatten_13/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_3/concat/axis?
-QNetwork/EncodingNetwork/concatenate_3/concatConcatV2AQNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape:output:07QNetwork/EncodingNetwork/flatten_13/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_3/concat?
)QNetwork/EncodingNetwork/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2+
)QNetwork/EncodingNetwork/flatten_14/Const?
+QNetwork/EncodingNetwork/flatten_14/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_3/concat:output:02QNetwork/EncodingNetwork/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2-
+QNetwork/EncodingNetwork/flatten_14/Reshape?
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_30/MatMulMatMul4QNetwork/EncodingNetwork/flatten_14/Reshape:output:0?QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_30/MatMul?
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_30/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_30/MatMul:product:0@QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_30/BiasAdd?
&QNetwork/EncodingNetwork/dense_30/ReluRelu2QNetwork/EncodingNetwork/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_30/Relu?
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_31/MatMulMatMul4QNetwork/EncodingNetwork/dense_30/Relu:activations:0?QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_31/MatMul?
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_31/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_31/MatMul:product:0@QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_31/BiasAdd?
&QNetwork/EncodingNetwork/dense_31/ReluRelu2QNetwork/EncodingNetwork/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_31/Relu?
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_32/MatMulMatMul4QNetwork/EncodingNetwork/dense_31/Relu:activations:0?QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_32/MatMul?
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_32/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_32/MatMul:product:0@QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_32/BiasAdd?
&QNetwork/EncodingNetwork/dense_32/ReluRelu2QNetwork/EncodingNetwork/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_32/Relu?
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_33/MatMulMatMul4QNetwork/EncodingNetwork/dense_32/Relu:activations:0?QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_33/MatMul?
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_33/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_33/MatMul:product:0@QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_33/BiasAdd?
&QNetwork/EncodingNetwork/dense_33/ReluRelu2QNetwork/EncodingNetwork/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_33/Relu?
'QNetwork/dense_34/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_34_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_34/MatMul/ReadVariableOp?
QNetwork/dense_34/MatMulMatMul4QNetwork/EncodingNetwork/dense_33/Relu:activations:0/QNetwork/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/MatMul?
(QNetwork/dense_34/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_34/BiasAdd/ReadVariableOp?
QNetwork/dense_34/BiasAddBiasAdd"QNetwork/dense_34/MatMul:product:00QNetwork/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_34/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
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
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp)^QNetwork/dense_34/BiasAdd/ReadVariableOp(^QNetwork/dense_34/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp2T
(QNetwork/dense_34/BiasAdd/ReadVariableOp(QNetwork/dense_34/BiasAdd/ReadVariableOp2R
'QNetwork/dense_34/MatMul/ReadVariableOp'QNetwork/dense_34/MatMul/ReadVariableOp:N J
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
?
?
,__inference_conv2d_3_layer_call_fn_167935543

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
GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1679351282
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
5
 
__inference_<lambda>_167934780*
_input_shapes 
?
)
'__inference_signature_wrapper_167935114?
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
GPU2*0J 8? *6
f1R/
-__inference_function_with_signature_1679351102
PartitionedCall*
_input_shapes 
??
?
+__inference_polymorphic_action_fn_167935372
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_board
time_step_observation_markQ
Mqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource4
0qnetwork_dense_34_matmul_readvariableop_resource5
1qnetwork_dense_34_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?(QNetwork/dense_34/BiasAdd/ReadVariableOp?'QNetwork/dense_34/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2DConv2Dtime_step_observation_boardLQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd?
6QNetwork/EncodingNetwork/sequential_3/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6QNetwork/EncodingNetwork/sequential_3/flatten_12/Const?
8QNetwork/EncodingNetwork/sequential_3/flatten_12/ReshapeReshape?QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd:output:0?QNetwork/EncodingNetwork/sequential_3/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2:
8QNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape?
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim?
.QNetwork/EncodingNetwork/flatten_13/ExpandDims
ExpandDimstime_step_observation_mark;QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????20
.QNetwork/EncodingNetwork/flatten_13/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_3/concat/axis?
-QNetwork/EncodingNetwork/concatenate_3/concatConcatV2AQNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape:output:07QNetwork/EncodingNetwork/flatten_13/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_3/concat?
)QNetwork/EncodingNetwork/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2+
)QNetwork/EncodingNetwork/flatten_14/Const?
+QNetwork/EncodingNetwork/flatten_14/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_3/concat:output:02QNetwork/EncodingNetwork/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2-
+QNetwork/EncodingNetwork/flatten_14/Reshape?
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_30/MatMulMatMul4QNetwork/EncodingNetwork/flatten_14/Reshape:output:0?QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_30/MatMul?
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_30/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_30/MatMul:product:0@QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_30/BiasAdd?
&QNetwork/EncodingNetwork/dense_30/ReluRelu2QNetwork/EncodingNetwork/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_30/Relu?
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_31/MatMulMatMul4QNetwork/EncodingNetwork/dense_30/Relu:activations:0?QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_31/MatMul?
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_31/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_31/MatMul:product:0@QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_31/BiasAdd?
&QNetwork/EncodingNetwork/dense_31/ReluRelu2QNetwork/EncodingNetwork/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_31/Relu?
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_32/MatMulMatMul4QNetwork/EncodingNetwork/dense_31/Relu:activations:0?QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_32/MatMul?
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_32/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_32/MatMul:product:0@QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_32/BiasAdd?
&QNetwork/EncodingNetwork/dense_32/ReluRelu2QNetwork/EncodingNetwork/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_32/Relu?
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_33/MatMulMatMul4QNetwork/EncodingNetwork/dense_32/Relu:activations:0?QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_33/MatMul?
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_33/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_33/MatMul:product:0@QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_33/BiasAdd?
&QNetwork/EncodingNetwork/dense_33/ReluRelu2QNetwork/EncodingNetwork/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_33/Relu?
'QNetwork/dense_34/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_34_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_34/MatMul/ReadVariableOp?
QNetwork/dense_34/MatMulMatMul4QNetwork/EncodingNetwork/dense_33/Relu:activations:0/QNetwork/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/MatMul?
(QNetwork/dense_34/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_34/BiasAdd/ReadVariableOp?
QNetwork/dense_34/BiasAddBiasAdd"QNetwork/dense_34/MatMul:product:00QNetwork/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_34/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
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
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp)^QNetwork/dense_34/BiasAdd/ReadVariableOp(^QNetwork/dense_34/MatMul/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????::::::::::::2t
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp2T
(QNetwork/dense_34/BiasAdd/ReadVariableOp(QNetwork/dense_34/BiasAdd/ReadVariableOp2R
'QNetwork/dense_34/MatMul/ReadVariableOp'QNetwork/dense_34/MatMul/ReadVariableOp:X T
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
?
9
'__inference_signature_wrapper_167935092

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
GPU2*0J 8? *6
f1R/
-__inference_function_with_signature_1679350872
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935182

inputs
conv2d_3_167935175
conv2d_3_167935177
identity?? conv2d_3/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_167935175conv2d_3_167935177*
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
GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1679351282"
 conv2d_3/StatefulPartitionedCall?
flatten_12/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *R
fMRK
I__inference_flatten_12_layer_call_and_return_conditional_losses_1679351502
flatten_12/PartitionedCall?
IdentityIdentity#flatten_12/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935452

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshapeconv2d_3/BiasAdd:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
IdentityIdentityflatten_12/Reshape:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
9
'__inference_get_initial_state_167935440

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
9
'__inference_get_initial_state_167935086

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
e
I__inference_flatten_12_layer_call_and_return_conditional_losses_167935549

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
?
J
.__inference_flatten_12_layer_call_fn_167935554

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
GPU2*0J 8? *R
fMRK
I__inference_flatten_12_layer_call_and_return_conditional_losses_1679351502
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
0__inference_sequential_3_layer_call_fn_167935515
conv2d_3_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0*
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
GPU2*0J 8? *T
fORM
K__inference_sequential_3_layer_call_and_return_conditional_losses_1679351822
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
_user_specified_nameconv2d_3_input
?
?
0__inference_sequential_3_layer_call_fn_167935473

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
GPU2*0J 8? *T
fORM
K__inference_sequential_3_layer_call_and_return_conditional_losses_1679351822
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
?
/
-__inference_function_with_signature_167935110?
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
GPU2*0J 8? *'
f"R 
__inference_<lambda>_1679347802
PartitionedCall*
_input_shapes 
?
?
-__inference_function_with_signature_167935045
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
GPU2*0J 8? *4
f/R-
+__inference_polymorphic_action_fn_1679350182
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
G__inference_conv2d_3_layer_call_and_return_conditional_losses_167935128

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
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935464

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAddu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshapeconv2d_3/BiasAdd:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
IdentityIdentityflatten_12/Reshape:output:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_3_layer_call_fn_167935524
conv2d_3_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0*
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
GPU2*0J 8? *T
fORM
K__inference_sequential_3_layer_call_and_return_conditional_losses_1679352012
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
_user_specified_nameconv2d_3_input
?
?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935201

inputs
conv2d_3_167935194
conv2d_3_167935196
identity?? conv2d_3/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_167935194conv2d_3_167935196*
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
GPU2*0J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1679351282"
 conv2d_3/StatefulPartitionedCall?
flatten_12/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *R
fMRK
I__inference_flatten_12_layer_call_and_return_conditional_losses_1679351502
flatten_12/PartitionedCall?
IdentityIdentity#flatten_12/PartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
?
1__inference_polymorphic_distribution_fn_167935437
	step_type

reward
discount
observation_board
observation_markQ
Mqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resourceR
Nqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource4
0qnetwork_dense_34_matmul_readvariableop_resource5
1qnetwork_dense_34_biasadd_readvariableop_resource
identity??8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?(QNetwork/dense_34/BiasAdd/ReadVariableOp?'QNetwork/dense_34/MatMul/ReadVariableOp?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02F
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp?
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2DConv2Dobservation_boardLQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
27
5QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp?
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAddBiasAdd>QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D:output:0MQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@28
6QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd?
6QNetwork/EncodingNetwork/sequential_3/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6QNetwork/EncodingNetwork/sequential_3/flatten_12/Const?
8QNetwork/EncodingNetwork/sequential_3/flatten_12/ReshapeReshape?QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd:output:0?QNetwork/EncodingNetwork/sequential_3/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2:
8QNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape?
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim?
.QNetwork/EncodingNetwork/flatten_13/ExpandDims
ExpandDimsobservation_mark;QNetwork/EncodingNetwork/flatten_13/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????20
.QNetwork/EncodingNetwork/flatten_13/ExpandDims?
2QNetwork/EncodingNetwork/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2QNetwork/EncodingNetwork/concatenate_3/concat/axis?
-QNetwork/EncodingNetwork/concatenate_3/concatConcatV2AQNetwork/EncodingNetwork/sequential_3/flatten_12/Reshape:output:07QNetwork/EncodingNetwork/flatten_13/ExpandDims:output:0;QNetwork/EncodingNetwork/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2/
-QNetwork/EncodingNetwork/concatenate_3/concat?
)QNetwork/EncodingNetwork/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2+
)QNetwork/EncodingNetwork/flatten_14/Const?
+QNetwork/EncodingNetwork/flatten_14/ReshapeReshape6QNetwork/EncodingNetwork/concatenate_3/concat:output:02QNetwork/EncodingNetwork/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2-
+QNetwork/EncodingNetwork/flatten_14/Reshape?
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_30/MatMulMatMul4QNetwork/EncodingNetwork/flatten_14/Reshape:output:0?QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_30/MatMul?
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_30/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_30/MatMul:product:0@QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_30/BiasAdd?
&QNetwork/EncodingNetwork/dense_30/ReluRelu2QNetwork/EncodingNetwork/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_30/Relu?
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_31/MatMulMatMul4QNetwork/EncodingNetwork/dense_30/Relu:activations:0?QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_31/MatMul?
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_31/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_31/MatMul:product:0@QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_31/BiasAdd?
&QNetwork/EncodingNetwork/dense_31/ReluRelu2QNetwork/EncodingNetwork/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_31/Relu?
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_32/MatMulMatMul4QNetwork/EncodingNetwork/dense_31/Relu:activations:0?QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_32/MatMul?
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_32/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_32/MatMul:product:0@QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_32/BiasAdd?
&QNetwork/EncodingNetwork/dense_32/ReluRelu2QNetwork/EncodingNetwork/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_32/Relu?
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_33/MatMulMatMul4QNetwork/EncodingNetwork/dense_32/Relu:activations:0?QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_33/MatMul?
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_33/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_33/MatMul:product:0@QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_33/BiasAdd?
&QNetwork/EncodingNetwork/dense_33/ReluRelu2QNetwork/EncodingNetwork/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_33/Relu?
'QNetwork/dense_34/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_34_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_34/MatMul/ReadVariableOp?
QNetwork/dense_34/MatMulMatMul4QNetwork/EncodingNetwork/dense_33/Relu:activations:0/QNetwork/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/MatMul?
(QNetwork/dense_34/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_34/BiasAdd/ReadVariableOp?
QNetwork/dense_34/BiasAddBiasAdd"QNetwork/dense_34/MatMul:product:00QNetwork/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_34/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_34/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
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
IdentityIdentityCategorical_1/mode/Cast:y:09^QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOpF^QNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp)^QNetwork/dense_34/BiasAdd/ReadVariableOp(^QNetwork/dense_34/MatMul/ReadVariableOp*
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
8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_30/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_30/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_31/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_31/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_32/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_32/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_33/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_33/MatMul/ReadVariableOp2?
EQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/sequential_3/conv2d_3/BiasAdd/ReadVariableOp2?
DQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOpDQNetwork/EncodingNetwork/sequential_3/conv2d_3/Conv2D/ReadVariableOp2T
(QNetwork/dense_34/BiasAdd/ReadVariableOp(QNetwork/dense_34/BiasAdd/ReadVariableOp2R
'QNetwork/dense_34/MatMul/ReadVariableOp'QNetwork/dense_34/MatMul/ReadVariableOp:N J
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
?
?
'__inference_signature_wrapper_167935080
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
GPU2*0J 8? *6
f1R/
-__inference_function_with_signature_1679350452
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
_user_specified_name0/step_type"?L
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
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:ђ
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
):'@2conv2d_3/kernel
:@2conv2d_3/bias
<::
??2(QNetwork/EncodingNetwork/dense_30/kernel
5:3?2&QNetwork/EncodingNetwork/dense_30/bias
<::
??2(QNetwork/EncodingNetwork/dense_31/kernel
5:3?2&QNetwork/EncodingNetwork/dense_31/bias
<::
??2(QNetwork/EncodingNetwork/dense_32/kernel
5:3?2&QNetwork/EncodingNetwork/dense_32/bias
<::
??2(QNetwork/EncodingNetwork/dense_33/kernel
5:3?2&QNetwork/EncodingNetwork/dense_33/bias
+:)	?2QNetwork/dense_34/kernel
$:"2QNetwork/dense_34/bias
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
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
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 768]}, {"class_name": "TensorShape", "items": [1, 1]}]}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 7, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}]}}}
?
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
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
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 769}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 769]}}
?

kernel
bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
?

kernel
bias
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
?

kernel
bias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1024]}}
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
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 7, 1]}}
?
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
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
+__inference_polymorphic_action_fn_167935290
+__inference_polymorphic_action_fn_167935372?
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
1__inference_polymorphic_distribution_fn_167935437?
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
'__inference_get_initial_state_167935440?
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
__inference_<lambda>_167934780"?
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
__inference_<lambda>_167934777"?
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
'__inference_signature_wrapper_167935080
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
'__inference_signature_wrapper_167935092
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
'__inference_signature_wrapper_167935107"?
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
'__inference_signature_wrapper_167935114"?
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
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935506
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935464
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935494
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935452?
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
0__inference_sequential_3_layer_call_fn_167935473
0__inference_sequential_3_layer_call_fn_167935515
0__inference_sequential_3_layer_call_fn_167935524
0__inference_sequential_3_layer_call_fn_167935482?
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
,__inference_conv2d_3_layer_call_fn_167935543?
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
G__inference_conv2d_3_layer_call_and_return_conditional_losses_167935534?
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
.__inference_flatten_12_layer_call_fn_167935554?
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
I__inference_flatten_12_layer_call_and_return_conditional_losses_167935549?
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
 =
__inference_<lambda>_167934777?

? 
? "? 	6
__inference_<lambda>_167934780?

? 
? "? ?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_167935534l	7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
,__inference_conv2d_3_layer_call_fn_167935543_	7?4
-?*
(?%
inputs?????????
? " ??????????@?
I__inference_flatten_12_layer_call_and_return_conditional_losses_167935549a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
.__inference_flatten_12_layer_call_fn_167935554T7?4
-?*
(?%
inputs?????????@
? "???????????T
'__inference_get_initial_state_167935440)"?
?
?

batch_size 
? "? ?
+__inference_polymorphic_action_fn_167935290?	
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
info? ?
+__inference_polymorphic_action_fn_167935372?	
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
info? ?
1__inference_polymorphic_distribution_fn_167935437?	
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
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935452m	??<
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
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935464m	??<
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
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935494u	G?D
=?:
0?-
conv2d_3_input?????????
p

 
? "&?#
?
0??????????
? ?
K__inference_sequential_3_layer_call_and_return_conditional_losses_167935506u	G?D
=?:
0?-
conv2d_3_input?????????
p 

 
? "&?#
?
0??????????
? ?
0__inference_sequential_3_layer_call_fn_167935473`	??<
5?2
(?%
inputs?????????
p

 
? "????????????
0__inference_sequential_3_layer_call_fn_167935482`	??<
5?2
(?%
inputs?????????
p 

 
? "????????????
0__inference_sequential_3_layer_call_fn_167935515h	G?D
=?:
0?-
conv2d_3_input?????????
p

 
? "????????????
0__inference_sequential_3_layer_call_fn_167935524h	G?D
=?:
0?-
conv2d_3_input?????????
p 

 
? "????????????
'__inference_signature_wrapper_167935080?	
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
action?????????b
'__inference_signature_wrapper_16793509270?-
? 
&?#
!

batch_size?

batch_size "? [
'__inference_signature_wrapper_1679351070?

? 
? "?

int64?
int64 	?
'__inference_signature_wrapper_167935114?

? 
? "? 