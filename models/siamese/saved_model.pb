??&
??
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718?? 
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
conv1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		`*
shared_nameconv1_1/kernel
y
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*&
_output_shapes
:		`*
dtype0
p
conv1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv1_1/bias
i
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
_output_shapes
:`*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:8*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:8*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:8*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:8*
dtype0
?
conv2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`?*
shared_nameconv2_1/kernel
z
"conv2_1/kernel/Read/ReadVariableOpReadVariableOpconv2_1/kernel*'
_output_shapes
:`?*
dtype0
q
conv2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2_1/bias
j
 conv2_1/bias/Read/ReadVariableOpReadVariableOpconv2_1/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
conv3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_1/kernel
{
"conv3_1/kernel/Read/ReadVariableOpReadVariableOpconv3_1/kernel*(
_output_shapes
:??*
dtype0
q
conv3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_1/bias
j
 conv3_1/bias/Read/ReadVariableOpReadVariableOpconv3_1/bias*
_output_shapes	
:?*
dtype0
?
conv3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_2/kernel
{
"conv3_2/kernel/Read/ReadVariableOpReadVariableOpconv3_2/kernel*(
_output_shapes
:??*
dtype0
q
conv3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_2/bias
j
 conv3_2/bias/Read/ReadVariableOpReadVariableOpconv3_2/bias*
_output_shapes	
:?*
dtype0
?
conv3_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3_3/kernel
{
"conv3_3/kernel/Read/ReadVariableOpReadVariableOpconv3_3/kernel*(
_output_shapes
:??*
dtype0
q
conv3_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_3/bias
j
 conv3_3/bias/Read/ReadVariableOpReadVariableOpconv3_3/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
RMSprop/conv1_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:		`*+
shared_nameRMSprop/conv1_1/kernel/rms
?
.RMSprop/conv1_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_1/kernel/rms*&
_output_shapes
:		`*
dtype0
?
RMSprop/conv1_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*)
shared_nameRMSprop/conv1_1/bias/rms
?
,RMSprop/conv1_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_1/bias/rms*
_output_shapes
:`*
dtype0
?
%RMSprop/batch_normalization/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*6
shared_name'%RMSprop/batch_normalization/gamma/rms
?
9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOpReadVariableOp%RMSprop/batch_normalization/gamma/rms*
_output_shapes
:8*
dtype0
?
$RMSprop/batch_normalization/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*5
shared_name&$RMSprop/batch_normalization/beta/rms
?
8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOpReadVariableOp$RMSprop/batch_normalization/beta/rms*
_output_shapes
:8*
dtype0
?
RMSprop/conv2_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`?*+
shared_nameRMSprop/conv2_1/kernel/rms
?
.RMSprop/conv2_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_1/kernel/rms*'
_output_shapes
:`?*
dtype0
?
RMSprop/conv2_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv2_1/bias/rms
?
,RMSprop/conv2_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_1/bias/rms*
_output_shapes	
:?*
dtype0
?
'RMSprop/batch_normalization_1/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'RMSprop/batch_normalization_1/gamma/rms
?
;RMSprop/batch_normalization_1/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_1/gamma/rms*
_output_shapes
:*
dtype0
?
&RMSprop/batch_normalization_1/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/batch_normalization_1/beta/rms
?
:RMSprop/batch_normalization_1/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_1/beta/rms*
_output_shapes
:*
dtype0
?
RMSprop/conv3_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameRMSprop/conv3_1/kernel/rms
?
.RMSprop/conv3_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_1/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv3_1/bias/rms
?
,RMSprop/conv3_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_1/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv3_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameRMSprop/conv3_2/kernel/rms
?
.RMSprop/conv3_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_2/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv3_2/bias/rms
?
,RMSprop/conv3_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_2/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv3_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameRMSprop/conv3_3/kernel/rms
?
.RMSprop/conv3_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_3/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv3_3/bias/rms
?
,RMSprop/conv3_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_3/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameRMSprop/dense/kernel/rms
?
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameRMSprop/dense/bias/rms
~
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameRMSprop/dense_1/kernel/rms
?
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/dense_1/bias/rms
?
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes	
:?*
dtype0

NoOpNoOp
?`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?_
value?_B?_ B?_
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
 
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer_with_weights-6
layer-8
layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
?
"iter
	#decay
$learning_rate
%momentum
&rho
'rms?
(rms?
)rms?
*rms?
-rms?
.rms?
/rms?
0rms?
3rms?
4rms?
5rms?
6rms?
7rms?
8rms?
9rms?
:rms?
;rms?
<rms?
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
 
?
'0
(1
)2
*3
-4
.5
/6
07
38
49
510
611
712
813
914
:15
;16
<17
?
	variables
=metrics
regularization_losses
>layer_regularization_losses

?layers
@non_trainable_variables
trainable_variables
Alayer_metrics
 
h

'kernel
(bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
?
Faxis
	)gamma
*beta
+moving_mean
,moving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
R
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
h

-kernel
.bias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
?
Saxis
	/gamma
0beta
1moving_mean
2moving_variance
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
R
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
h

3kernel
4bias
\	variables
]regularization_losses
^trainable_variables
_	keras_api
h

5kernel
6bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
h

7kernel
8bias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
R
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
h

9kernel
:bias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
R
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
h

;kernel
<bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
 
?
'0
(1
)2
*3
-4
.5
/6
07
38
49
510
611
712
813
914
:15
;16
<17
?
	variables
?metrics
regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
trainable_variables
?layer_metrics
 
 
 
?
	variables
?metrics
regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
 trainable_variables
?layer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv1_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv1_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv3_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv3_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv3_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv3_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv3_3/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv3_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

0
1
2
3

+0
,1
12
23
 

'0
(1
 

'0
(1
?
B	variables
?metrics
Cregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Dtrainable_variables
?layer_metrics
 

)0
*1
+2
,3
 

)0
*1
?
G	variables
?metrics
Hregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Itrainable_variables
?layer_metrics
 
 
 
?
K	variables
?metrics
Lregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Mtrainable_variables
?layer_metrics

-0
.1
 

-0
.1
?
O	variables
?metrics
Pregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Qtrainable_variables
?layer_metrics
 

/0
01
12
23
 

/0
01
?
T	variables
?metrics
Uregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Vtrainable_variables
?layer_metrics
 
 
 
?
X	variables
?metrics
Yregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Ztrainable_variables
?layer_metrics

30
41
 

30
41
?
\	variables
?metrics
]regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
^trainable_variables
?layer_metrics

50
61
 

50
61
?
`	variables
?metrics
aregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
btrainable_variables
?layer_metrics

70
81
 

70
81
?
d	variables
?metrics
eregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ftrainable_variables
?layer_metrics
 
 
 
?
h	variables
?metrics
iregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
jtrainable_variables
?layer_metrics
 
 
 
?
l	variables
?metrics
mregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ntrainable_variables
?layer_metrics
 
 
 
?
p	variables
?metrics
qregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
rtrainable_variables
?layer_metrics

90
:1
 

90
:1
?
t	variables
?metrics
uregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
vtrainable_variables
?layer_metrics
 
 
 
?
x	variables
?metrics
yregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ztrainable_variables
?layer_metrics

;0
<1
 

;0
<1
?
|	variables
?metrics
}regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
~trainable_variables
?layer_metrics
 
 
n
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14

+0
,1
12
23
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 

+0
,1
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

10
21
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
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
tr
VARIABLE_VALUERMSprop/conv1_1/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUERMSprop/conv1_1/bias/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%RMSprop/batch_normalization/gamma/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$RMSprop/batch_normalization/beta/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUERMSprop/conv2_1/kernel/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUERMSprop/conv2_1/bias/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE'RMSprop/batch_normalization_1/gamma/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE&RMSprop/batch_normalization_1/beta/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUERMSprop/conv3_1/kernel/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/conv3_1/bias/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUERMSprop/conv3_2/kernel/rmsEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/conv3_2/bias/rmsEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUERMSprop/conv3_3/kernel/rmsEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/conv3_3/bias/rmsEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/dense/kernel/rmsEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUERMSprop/dense/bias/rmsEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUERMSprop/dense_1/kernel/rmsEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/dense_1/bias/rmsEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv1_1/kernelconv1_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2_1/kernelconv2_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3_1/kernelconv3_1/biasconv3_2/kernelconv3_2/biasconv3_3/kernelconv3_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_35514282
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"conv2_1/kernel/Read/ReadVariableOp conv2_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"conv3_1/kernel/Read/ReadVariableOp conv3_1/bias/Read/ReadVariableOp"conv3_2/kernel/Read/ReadVariableOp conv3_2/bias/Read/ReadVariableOp"conv3_3/kernel/Read/ReadVariableOp conv3_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.RMSprop/conv1_1/kernel/rms/Read/ReadVariableOp,RMSprop/conv1_1/bias/rms/Read/ReadVariableOp9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOp8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOp.RMSprop/conv2_1/kernel/rms/Read/ReadVariableOp,RMSprop/conv2_1/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_1/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_1/beta/rms/Read/ReadVariableOp.RMSprop/conv3_1/kernel/rms/Read/ReadVariableOp,RMSprop/conv3_1/bias/rms/Read/ReadVariableOp.RMSprop/conv3_2/kernel/rms/Read/ReadVariableOp,RMSprop/conv3_2/bias/rms/Read/ReadVariableOp.RMSprop/conv3_3/kernel/rms/Read/ReadVariableOp,RMSprop/conv3_3/bias/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOpConst*>
Tin7
523	*
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
!__inference__traced_save_35516045
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoconv1_1/kernelconv1_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2_1/kernelconv2_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3_1/kernelconv3_1/biasconv3_2/kernelconv3_2/biasconv3_3/kernelconv3_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcounttotal_1count_1RMSprop/conv1_1/kernel/rmsRMSprop/conv1_1/bias/rms%RMSprop/batch_normalization/gamma/rms$RMSprop/batch_normalization/beta/rmsRMSprop/conv2_1/kernel/rmsRMSprop/conv2_1/bias/rms'RMSprop/batch_normalization_1/gamma/rms&RMSprop/batch_normalization_1/beta/rmsRMSprop/conv3_1/kernel/rmsRMSprop/conv3_1/bias/rmsRMSprop/conv3_2/kernel/rmsRMSprop/conv3_2/bias/rmsRMSprop/conv3_3/kernel/rmsRMSprop/conv3_3/bias/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rms*=
Tin6
422*
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
$__inference__traced_restore_35516202??
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35512247

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+?????????8??????????????????:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35512759

inputs*
conv1_1_35512494:		`
conv1_1_35512496:`*
batch_normalization_35512517:8*
batch_normalization_35512519:8*
batch_normalization_35512521:8*
batch_normalization_35512523:8+
conv2_1_35512545:`?
conv2_1_35512547:	?,
batch_normalization_1_35512568:,
batch_normalization_1_35512570:,
batch_normalization_1_35512572:,
batch_normalization_1_35512574:,
conv3_1_35512596:??
conv3_1_35512598:	?,
conv3_2_35512619:??
conv3_2_35512621:	?,
conv3_3_35512642:??
conv3_3_35512644:	?"
dense_35512681:
??
dense_35512683:	?$
dense_1_35512711:
??
dense_1_35512713:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/StatefulPartitionedCall?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/StatefulPartitionedCall?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/StatefulPartitionedCall?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/StatefulPartitionedCall?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_1_35512494conv1_1_35512496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1_1_layer_call_and_return_conditional_losses_355124932!
conv1_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0batch_normalization_35512517batch_normalization_35512519batch_normalization_35512521batch_normalization_35512523*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355125162-
+batch_normalization/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_355123132
max_pooling2d/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_1_35512545conv2_1_35512547*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2_1_layer_call_and_return_conditional_losses_355125442!
conv2_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0batch_normalization_1_35512568batch_normalization_1_35512570batch_normalization_1_35512572batch_normalization_1_35512574*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355125672/
-batch_normalization_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_355124512!
max_pooling2d_1/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_1_35512596conv3_1_35512598*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_1_layer_call_and_return_conditional_losses_355125952!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_35512619conv3_2_35512621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_2_layer_call_and_return_conditional_losses_355126182!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_35512642conv3_3_35512644*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_3_layer_call_and_return_conditional_losses_355126412!
conv3_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_355124632!
max_pooling2d_2/PartitionedCall?
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355126532
dropout/PartitionedCall?
Flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Flatten_layer_call_and_return_conditional_losses_355126612
Flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_35512681dense_35512683*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_355126802
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355126912
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_35512711dense_1_35512713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_355127102!
dense_1/StatefulPartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_1_35512494*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_1_35512545*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_1_35512596*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_2_35512619*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_3_35512642*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35512681* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_35512711* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall1^conv1_1/kernel/Regularizer/Square/ReadVariableOp ^conv2_1/StatefulPartitionedCall1^conv2_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_1/StatefulPartitionedCall1^conv3_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_2/StatefulPartitionedCall1^conv3_2/kernel/Regularizer/Square/ReadVariableOp ^conv3_3/StatefulPartitionedCall1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35515126

inputs@
&conv1_1_conv2d_readvariableop_resource:		`5
'conv1_1_biasadd_readvariableop_resource:`9
+batch_normalization_readvariableop_resource:8;
-batch_normalization_readvariableop_1_resource:8J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:8L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:8A
&conv2_1_conv2d_readvariableop_resource:`?6
'conv2_1_biasadd_readvariableop_resource:	?;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:B
&conv3_1_conv2d_readvariableop_resource:??6
'conv3_1_biasadd_readvariableop_resource:	?B
&conv3_2_conv2d_readvariableop_resource:??6
'conv3_2_biasadd_readvariableop_resource:	?B
&conv3_3_conv2d_readvariableop_resource:??6
'conv3_3_biasadd_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/BiasAdd/ReadVariableOp?conv2_1/Conv2D/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/BiasAdd/ReadVariableOp?conv3_1/Conv2D/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/BiasAdd/ReadVariableOp?conv3_2/Conv2D/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/BiasAdd/ReadVariableOp?conv3_3/Conv2D/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dinputs%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
conv1_1/BiasAddx
conv1_1/ReluReluconv1_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
conv1_1/Relu?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv1_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2_1/Conv2D/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02
conv2_1/Conv2D/ReadVariableOp?
conv2_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0%conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2_1/Conv2D?
conv2_1/BiasAdd/ReadVariableOpReadVariableOp'conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_1/BiasAdd/ReadVariableOp?
conv2_1/BiasAddBiasAddconv2_1/Conv2D:output:0&conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2_1/BiasAddy
conv2_1/ReluReluconv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2_1/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv3_1/Conv2D/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_1/Conv2D/ReadVariableOp?
conv3_1/Conv2DConv2D max_pooling2d_1/MaxPool:output:0%conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv3_1/Conv2D?
conv3_1/BiasAdd/ReadVariableOpReadVariableOp'conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_1/BiasAdd/ReadVariableOp?
conv3_1/BiasAddBiasAddconv3_1/Conv2D:output:0&conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv3_1/BiasAddy
conv3_1/ReluReluconv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
conv3_1/Relu?
conv3_2/Conv2D/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_2/Conv2D/ReadVariableOp?
conv3_2/Conv2DConv2Dconv3_1/Relu:activations:0%conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3_2/Conv2D?
conv3_2/BiasAdd/ReadVariableOpReadVariableOp'conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_2/BiasAdd/ReadVariableOp?
conv3_2/BiasAddBiasAddconv3_2/Conv2D:output:0&conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_2/BiasAddy
conv3_2/ReluReluconv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_2/Relu?
conv3_3/Conv2D/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_3/Conv2D/ReadVariableOp?
conv3_3/Conv2DConv2Dconv3_2/Relu:activations:0%conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3_3/Conv2D?
conv3_3/BiasAdd/ReadVariableOpReadVariableOp'conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_3/BiasAdd/ReadVariableOp?
conv3_3/BiasAddBiasAddconv3_3/Conv2D:output:0&conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_3/BiasAddy
conv3_3/ReluReluconv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_3/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMul max_pooling2d_2/MaxPool:output:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul~
dropout/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/dropout/Mul_1o
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Flatten/Const?
Flatten/ReshapeReshapedropout/dropout/Mul_1:z:0Flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
Flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulFlatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitydense_1/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp1^conv1_1/kernel/Regularizer/Square/ReadVariableOp^conv2_1/BiasAdd/ReadVariableOp^conv2_1/Conv2D/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp^conv3_1/BiasAdd/ReadVariableOp^conv3_1/Conv2D/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp^conv3_2/BiasAdd/ReadVariableOp^conv3_2/Conv2D/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp^conv3_3/BiasAdd/ReadVariableOp^conv3_3/Conv2D/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2@
conv2_1/BiasAdd/ReadVariableOpconv2_1/BiasAdd/ReadVariableOp2>
conv2_1/Conv2D/ReadVariableOpconv2_1/Conv2D/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2@
conv3_1/BiasAdd/ReadVariableOpconv3_1/BiasAdd/ReadVariableOp2>
conv3_1/Conv2D/ReadVariableOpconv3_1/Conv2D/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2@
conv3_2/BiasAdd/ReadVariableOpconv3_2/BiasAdd/ReadVariableOp2>
conv3_2/Conv2D/ReadVariableOpconv3_2/Conv2D/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2@
conv3_3/BiasAdd/ReadVariableOpconv3_3/BiasAdd/ReadVariableOp2>
conv3_3/Conv2D/ReadVariableOpconv3_3/Conv2D/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?!
C__inference_model_layer_call_and_return_conditional_losses_35514482
inputs_0
inputs_1K
1sequential_conv1_1_conv2d_readvariableop_resource:		`@
2sequential_conv1_1_biasadd_readvariableop_resource:`D
6sequential_batch_normalization_readvariableop_resource:8F
8sequential_batch_normalization_readvariableop_1_resource:8U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:8W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:8L
1sequential_conv2_1_conv2d_readvariableop_resource:`?A
2sequential_conv2_1_biasadd_readvariableop_resource:	?F
8sequential_batch_normalization_1_readvariableop_resource:H
:sequential_batch_normalization_1_readvariableop_1_resource:W
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:Y
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:M
1sequential_conv3_1_conv2d_readvariableop_resource:??A
2sequential_conv3_1_biasadd_readvariableop_resource:	?M
1sequential_conv3_2_conv2d_readvariableop_resource:??A
2sequential_conv3_2_biasadd_readvariableop_resource:	?M
1sequential_conv3_3_conv2d_readvariableop_resource:??A
2sequential_conv3_3_biasadd_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?/sequential/batch_normalization/ReadVariableOp_2?/sequential/batch_normalization/ReadVariableOp_3?@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?/sequential/batch_normalization_1/ReadVariableOp?1sequential/batch_normalization_1/ReadVariableOp_1?1sequential/batch_normalization_1/ReadVariableOp_2?1sequential/batch_normalization_1/ReadVariableOp_3?)sequential/conv1_1/BiasAdd/ReadVariableOp?+sequential/conv1_1/BiasAdd_1/ReadVariableOp?(sequential/conv1_1/Conv2D/ReadVariableOp?*sequential/conv1_1/Conv2D_1/ReadVariableOp?)sequential/conv2_1/BiasAdd/ReadVariableOp?+sequential/conv2_1/BiasAdd_1/ReadVariableOp?(sequential/conv2_1/Conv2D/ReadVariableOp?*sequential/conv2_1/Conv2D_1/ReadVariableOp?)sequential/conv3_1/BiasAdd/ReadVariableOp?+sequential/conv3_1/BiasAdd_1/ReadVariableOp?(sequential/conv3_1/Conv2D/ReadVariableOp?*sequential/conv3_1/Conv2D_1/ReadVariableOp?)sequential/conv3_2/BiasAdd/ReadVariableOp?+sequential/conv3_2/BiasAdd_1/ReadVariableOp?(sequential/conv3_2/Conv2D/ReadVariableOp?*sequential/conv3_2/Conv2D_1/ReadVariableOp?)sequential/conv3_3/BiasAdd/ReadVariableOp?+sequential/conv3_3/BiasAdd_1/ReadVariableOp?(sequential/conv3_3/Conv2D/ReadVariableOp?*sequential/conv3_3/Conv2D_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/BiasAdd_1/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?(sequential/dense/MatMul_1/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/BiasAdd_1/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?*sequential/dense_1/MatMul_1/ReadVariableOp?
(sequential/conv1_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02*
(sequential/conv1_1/Conv2D/ReadVariableOp?
sequential/conv1_1/Conv2DConv2Dinputs_00sequential/conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
sequential/conv1_1/Conv2D?
)sequential/conv1_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02+
)sequential/conv1_1/BiasAdd/ReadVariableOp?
sequential/conv1_1/BiasAddBiasAdd"sequential/conv1_1/Conv2D:output:01sequential/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/BiasAdd?
sequential/conv1_1/ReluRelu#sequential/conv1_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/Relu?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3%sequential/conv1_1/Relu:activations:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
(sequential/conv2_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02*
(sequential/conv2_1/Conv2D/ReadVariableOp?
sequential/conv2_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:00sequential/conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv2_1/Conv2D?
)sequential/conv2_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv2_1/BiasAdd/ReadVariableOp?
sequential/conv2_1/BiasAddBiasAdd"sequential/conv2_1/Conv2D:output:01sequential/conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/BiasAdd?
sequential/conv2_1/ReluRelu#sequential/conv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/Relu?
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization_1/ReadVariableOp?
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%sequential/conv2_1/Relu:activations:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3?
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
(sequential/conv3_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_1/Conv2D/ReadVariableOp?
sequential/conv3_1/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:00sequential/conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
sequential/conv3_1/Conv2D?
)sequential/conv3_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_1/BiasAdd/ReadVariableOp?
sequential/conv3_1/BiasAddBiasAdd"sequential/conv3_1/Conv2D:output:01sequential/conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/BiasAdd?
sequential/conv3_1/ReluRelu#sequential/conv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/Relu?
(sequential/conv3_2/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_2/Conv2D/ReadVariableOp?
sequential/conv3_2/Conv2DConv2D%sequential/conv3_1/Relu:activations:00sequential/conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_2/Conv2D?
)sequential/conv3_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_2/BiasAdd/ReadVariableOp?
sequential/conv3_2/BiasAddBiasAdd"sequential/conv3_2/Conv2D:output:01sequential/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/BiasAdd?
sequential/conv3_2/ReluRelu#sequential/conv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/Relu?
(sequential/conv3_3/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_3/Conv2D/ReadVariableOp?
sequential/conv3_3/Conv2DConv2D%sequential/conv3_2/Relu:activations:00sequential/conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_3/Conv2D?
)sequential/conv3_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_3/BiasAdd/ReadVariableOp?
sequential/conv3_3/BiasAddBiasAdd"sequential/conv3_3/Conv2D:output:01sequential/conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/BiasAdd?
sequential/conv3_3/ReluRelu#sequential/conv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/Relu?
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/conv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool?
sequential/dropout/IdentityIdentity+sequential/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
sequential/dropout/Identity?
sequential/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/Flatten/Const?
sequential/Flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/Flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/Flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/Flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
sequential/dropout_1/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
sequential/dropout_1/Identity?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
*sequential/conv1_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02,
*sequential/conv1_1/Conv2D_1/ReadVariableOp?
sequential/conv1_1/Conv2D_1Conv2Dinputs_12sequential/conv1_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
sequential/conv1_1/Conv2D_1?
+sequential/conv1_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02-
+sequential/conv1_1/BiasAdd_1/ReadVariableOp?
sequential/conv1_1/BiasAdd_1BiasAdd$sequential/conv1_1/Conv2D_1:output:03sequential/conv1_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/BiasAdd_1?
sequential/conv1_1/Relu_1Relu%sequential/conv1_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/Relu_1?
/sequential/batch_normalization/ReadVariableOp_2ReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_2?
/sequential/batch_normalization/ReadVariableOp_3ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_3?
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?
1sequential/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3'sequential/conv1_1/Relu_1:activations:07sequential/batch_normalization/ReadVariableOp_2:value:07sequential/batch_normalization/ReadVariableOp_3:value:0Hsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Jsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 23
1sequential/batch_normalization/FusedBatchNormV3_1?
"sequential/max_pooling2d/MaxPool_1MaxPool5sequential/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d/MaxPool_1?
*sequential/conv2_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02,
*sequential/conv2_1/Conv2D_1/ReadVariableOp?
sequential/conv2_1/Conv2D_1Conv2D+sequential/max_pooling2d/MaxPool_1:output:02sequential/conv2_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv2_1/Conv2D_1?
+sequential/conv2_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv2_1/BiasAdd_1/ReadVariableOp?
sequential/conv2_1/BiasAdd_1BiasAdd$sequential/conv2_1/Conv2D_1:output:03sequential/conv2_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/BiasAdd_1?
sequential/conv2_1/Relu_1Relu%sequential/conv2_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/Relu_1?
1sequential/batch_normalization_1/ReadVariableOp_2ReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_2?
1sequential/batch_normalization_1/ReadVariableOp_3ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_3?
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?
3sequential/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3'sequential/conv2_1/Relu_1:activations:09sequential/batch_normalization_1/ReadVariableOp_2:value:09sequential/batch_normalization_1/ReadVariableOp_3:value:0Jsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Lsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 25
3sequential/batch_normalization_1/FusedBatchNormV3_1?
$sequential/max_pooling2d_1/MaxPool_1MaxPool7sequential/batch_normalization_1/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2&
$sequential/max_pooling2d_1/MaxPool_1?
*sequential/conv3_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_1/Conv2D_1/ReadVariableOp?
sequential/conv3_1/Conv2D_1Conv2D-sequential/max_pooling2d_1/MaxPool_1:output:02sequential/conv3_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
sequential/conv3_1/Conv2D_1?
+sequential/conv3_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_1/BiasAdd_1/ReadVariableOp?
sequential/conv3_1/BiasAdd_1BiasAdd$sequential/conv3_1/Conv2D_1:output:03sequential/conv3_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/BiasAdd_1?
sequential/conv3_1/Relu_1Relu%sequential/conv3_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/Relu_1?
*sequential/conv3_2/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_2/Conv2D_1/ReadVariableOp?
sequential/conv3_2/Conv2D_1Conv2D'sequential/conv3_1/Relu_1:activations:02sequential/conv3_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_2/Conv2D_1?
+sequential/conv3_2/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_2/BiasAdd_1/ReadVariableOp?
sequential/conv3_2/BiasAdd_1BiasAdd$sequential/conv3_2/Conv2D_1:output:03sequential/conv3_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/BiasAdd_1?
sequential/conv3_2/Relu_1Relu%sequential/conv3_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/Relu_1?
*sequential/conv3_3/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_3/Conv2D_1/ReadVariableOp?
sequential/conv3_3/Conv2D_1Conv2D'sequential/conv3_2/Relu_1:activations:02sequential/conv3_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_3/Conv2D_1?
+sequential/conv3_3/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_3/BiasAdd_1/ReadVariableOp?
sequential/conv3_3/BiasAdd_1BiasAdd$sequential/conv3_3/Conv2D_1:output:03sequential/conv3_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/BiasAdd_1?
sequential/conv3_3/Relu_1Relu%sequential/conv3_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/Relu_1?
$sequential/max_pooling2d_2/MaxPool_1MaxPool'sequential/conv3_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2&
$sequential/max_pooling2d_2/MaxPool_1?
sequential/dropout/Identity_1Identity-sequential/max_pooling2d_2/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/dropout/Identity_1?
sequential/Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/Flatten/Const_1?
sequential/Flatten/Reshape_1Reshape&sequential/dropout/Identity_1:output:0#sequential/Flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/Flatten/Reshape_1?
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp?
sequential/dense/MatMul_1MatMul%sequential/Flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul_1?
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp?
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd_1?
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu_1?
sequential/dropout_1/Identity_1Identity%sequential/dense/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2!
sequential/dropout_1/Identity_1?
*sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential/dense_1/MatMul_1/ReadVariableOp?
sequential/dense_1/MatMul_1MatMul(sequential/dropout_1/Identity_1:output:02sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul_1?
+sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/dense_1/BiasAdd_1/ReadVariableOp?
sequential/dense_1/BiasAdd_1BiasAdd%sequential/dense_1/MatMul_1:product:03sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd_1?
sequential/dense_1/Relu_1Relu%sequential/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu_1?

lambda/subSub%sequential/dense_1/Relu:activations:0'sequential/dense_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2

lambda/subk
lambda/SquareSquarelambda/sub:z:0*
T0*(
_output_shapes
:??????????2
lambda/Square~
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Sum/reduction_indices?

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

lambda/Suma
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/Const?
lambda/MaximumMaximumlambda/Sum:output:0lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximumh
lambda/SqrtSqrtlambda/Maximum:z:0*
T0*'
_output_shapes
:?????????2
lambda/Sqrt?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/Sqrt:y:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1A^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpC^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_10^sequential/batch_normalization/ReadVariableOp_20^sequential/batch_normalization/ReadVariableOp_3A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1C^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpE^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_12^sequential/batch_normalization_1/ReadVariableOp_22^sequential/batch_normalization_1/ReadVariableOp_3*^sequential/conv1_1/BiasAdd/ReadVariableOp,^sequential/conv1_1/BiasAdd_1/ReadVariableOp)^sequential/conv1_1/Conv2D/ReadVariableOp+^sequential/conv1_1/Conv2D_1/ReadVariableOp*^sequential/conv2_1/BiasAdd/ReadVariableOp,^sequential/conv2_1/BiasAdd_1/ReadVariableOp)^sequential/conv2_1/Conv2D/ReadVariableOp+^sequential/conv2_1/Conv2D_1/ReadVariableOp*^sequential/conv3_1/BiasAdd/ReadVariableOp,^sequential/conv3_1/BiasAdd_1/ReadVariableOp)^sequential/conv3_1/Conv2D/ReadVariableOp+^sequential/conv3_1/Conv2D_1/ReadVariableOp*^sequential/conv3_2/BiasAdd/ReadVariableOp,^sequential/conv3_2/BiasAdd_1/ReadVariableOp)^sequential/conv3_2/Conv2D/ReadVariableOp+^sequential/conv3_2/Conv2D_1/ReadVariableOp*^sequential/conv3_3/BiasAdd/ReadVariableOp,^sequential/conv3_3/BiasAdd_1/ReadVariableOp)^sequential/conv3_3/Conv2D/ReadVariableOp+^sequential/conv3_3/Conv2D_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/BiasAdd_1/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp+^sequential/dense_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2?
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12b
/sequential/batch_normalization/ReadVariableOp_2/sequential/batch_normalization/ReadVariableOp_22b
/sequential/batch_normalization/ReadVariableOp_3/sequential/batch_normalization/ReadVariableOp_32?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpBsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12f
1sequential/batch_normalization_1/ReadVariableOp_21sequential/batch_normalization_1/ReadVariableOp_22f
1sequential/batch_normalization_1/ReadVariableOp_31sequential/batch_normalization_1/ReadVariableOp_32V
)sequential/conv1_1/BiasAdd/ReadVariableOp)sequential/conv1_1/BiasAdd/ReadVariableOp2Z
+sequential/conv1_1/BiasAdd_1/ReadVariableOp+sequential/conv1_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv1_1/Conv2D/ReadVariableOp(sequential/conv1_1/Conv2D/ReadVariableOp2X
*sequential/conv1_1/Conv2D_1/ReadVariableOp*sequential/conv1_1/Conv2D_1/ReadVariableOp2V
)sequential/conv2_1/BiasAdd/ReadVariableOp)sequential/conv2_1/BiasAdd/ReadVariableOp2Z
+sequential/conv2_1/BiasAdd_1/ReadVariableOp+sequential/conv2_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv2_1/Conv2D/ReadVariableOp(sequential/conv2_1/Conv2D/ReadVariableOp2X
*sequential/conv2_1/Conv2D_1/ReadVariableOp*sequential/conv2_1/Conv2D_1/ReadVariableOp2V
)sequential/conv3_1/BiasAdd/ReadVariableOp)sequential/conv3_1/BiasAdd/ReadVariableOp2Z
+sequential/conv3_1/BiasAdd_1/ReadVariableOp+sequential/conv3_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_1/Conv2D/ReadVariableOp(sequential/conv3_1/Conv2D/ReadVariableOp2X
*sequential/conv3_1/Conv2D_1/ReadVariableOp*sequential/conv3_1/Conv2D_1/ReadVariableOp2V
)sequential/conv3_2/BiasAdd/ReadVariableOp)sequential/conv3_2/BiasAdd/ReadVariableOp2Z
+sequential/conv3_2/BiasAdd_1/ReadVariableOp+sequential/conv3_2/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_2/Conv2D/ReadVariableOp(sequential/conv3_2/Conv2D/ReadVariableOp2X
*sequential/conv3_2/Conv2D_1/ReadVariableOp*sequential/conv3_2/Conv2D_1/ReadVariableOp2V
)sequential/conv3_3/BiasAdd/ReadVariableOp)sequential/conv3_3/BiasAdd/ReadVariableOp2Z
+sequential/conv3_3/BiasAdd_1/ReadVariableOp+sequential/conv3_3/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_3/Conv2D/ReadVariableOp(sequential/conv3_3/Conv2D/ReadVariableOp2X
*sequential/conv3_3/Conv2D_1/ReadVariableOp*sequential/conv3_3/Conv2D_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/BiasAdd_1/ReadVariableOp+sequential/dense_1/BiasAdd_1/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2X
*sequential/dense_1/MatMul_1/ReadVariableOp*sequential/dense_1/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?
?
E__inference_conv3_3_layer_call_and_return_conditional_losses_35515659

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?d
?

C__inference_model_layer_call_and_return_conditional_losses_35514182
input_1
input_2-
sequential_35514070:		`!
sequential_35514072:`!
sequential_35514074:8!
sequential_35514076:8!
sequential_35514078:8!
sequential_35514080:8.
sequential_35514082:`?"
sequential_35514084:	?!
sequential_35514086:!
sequential_35514088:!
sequential_35514090:!
sequential_35514092:/
sequential_35514094:??"
sequential_35514096:	?/
sequential_35514098:??"
sequential_35514100:	?/
sequential_35514102:??"
sequential_35514104:	?'
sequential_35514106:
??"
sequential_35514108:	?'
sequential_35514110:
??"
sequential_35514112:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_35514070sequential_35514072sequential_35514074sequential_35514076sequential_35514078sequential_35514080sequential_35514082sequential_35514084sequential_35514086sequential_35514088sequential_35514090sequential_35514092sequential_35514094sequential_35514096sequential_35514098sequential_35514100sequential_35514102sequential_35514104sequential_35514106sequential_35514108sequential_35514110sequential_35514112*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722$
"sequential/StatefulPartitionedCall?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_35514070sequential_35514072sequential_35514074sequential_35514076sequential_35514078sequential_35514080sequential_35514082sequential_35514084sequential_35514086sequential_35514088sequential_35514090sequential_35514092sequential_35514094sequential_35514096sequential_35514098sequential_35514100sequential_35514102sequential_35514104sequential_35514106sequential_35514108sequential_35514110sequential_35514112#^sequential/StatefulPartitionedCall*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722&
$sequential/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355136812
lambda/PartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514070*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514082*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514094*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514098*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514102*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514106* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35514110* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/PartitionedCall:output:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
a
E__inference_Flatten_layer_call_and_return_conditional_losses_35515701

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_1_layer_call_fn_35515765

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355128362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
n
D__inference_lambda_layer_call_and_return_conditional_losses_35513567

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35513172

inputs*
conv1_1_35513070:		`
conv1_1_35513072:`*
batch_normalization_35513075:8*
batch_normalization_35513077:8*
batch_normalization_35513079:8*
batch_normalization_35513081:8+
conv2_1_35513085:`?
conv2_1_35513087:	?,
batch_normalization_1_35513090:,
batch_normalization_1_35513092:,
batch_normalization_1_35513094:,
batch_normalization_1_35513096:,
conv3_1_35513100:??
conv3_1_35513102:	?,
conv3_2_35513105:??
conv3_2_35513107:	?,
conv3_3_35513110:??
conv3_3_35513112:	?"
dense_35513118:
??
dense_35513120:	?$
dense_1_35513124:
??
dense_1_35513126:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/StatefulPartitionedCall?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/StatefulPartitionedCall?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/StatefulPartitionedCall?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/StatefulPartitionedCall?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_1_35513070conv1_1_35513072*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1_1_layer_call_and_return_conditional_losses_355124932!
conv1_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0batch_normalization_35513075batch_normalization_35513077batch_normalization_35513079batch_normalization_35513081*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355129952-
+batch_normalization/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_355123132
max_pooling2d/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_1_35513085conv2_1_35513087*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2_1_layer_call_and_return_conditional_losses_355125442!
conv2_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0batch_normalization_1_35513090batch_normalization_1_35513092batch_normalization_1_35513094batch_normalization_1_35513096*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355129412/
-batch_normalization_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_355124512!
max_pooling2d_1/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_1_35513100conv3_1_35513102*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_1_layer_call_and_return_conditional_losses_355125952!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_35513105conv3_2_35513107*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_2_layer_call_and_return_conditional_losses_355126182!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_35513110conv3_3_35513112*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_3_layer_call_and_return_conditional_losses_355126412!
conv3_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_355124632!
max_pooling2d_2/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355128752!
dropout/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Flatten_layer_call_and_return_conditional_losses_355126612
Flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_35513118dense_35513120*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_355126802
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355128362#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_35513124dense_1_35513126*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_355127102!
dense_1/StatefulPartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_1_35513070*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_1_35513085*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_1_35513100*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_2_35513105*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_3_35513110*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35513118* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_35513124* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall1^conv1_1/kernel/Regularizer/Square/ReadVariableOp ^conv2_1/StatefulPartitionedCall1^conv2_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_1/StatefulPartitionedCall1^conv3_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_2/StatefulPartitionedCall1^conv3_2/kernel/Regularizer/Square/ReadVariableOp ^conv3_3/StatefulPartitionedCall1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_35512806
conv1_1_input!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????@@
'
_user_specified_nameconv1_1_input
?
?
8__inference_batch_normalization_1_layer_call_fn_35515533

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355123412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35512995

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
?
*__inference_conv1_1_layer_call_fn_35515292

inputs!
unknown:		`
	unknown_0:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1_1_layer_call_and_return_conditional_losses_355124932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
E__inference_conv3_3_layer_call_and_return_conditional_losses_35512641

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?d
?

C__inference_model_layer_call_and_return_conditional_losses_35513853

inputs
inputs_1-
sequential_35513741:		`!
sequential_35513743:`!
sequential_35513745:8!
sequential_35513747:8!
sequential_35513749:8!
sequential_35513751:8.
sequential_35513753:`?"
sequential_35513755:	?!
sequential_35513757:!
sequential_35513759:!
sequential_35513761:!
sequential_35513763:/
sequential_35513765:??"
sequential_35513767:	?/
sequential_35513769:??"
sequential_35513771:	?/
sequential_35513773:??"
sequential_35513775:	?'
sequential_35513777:
??"
sequential_35513779:	?'
sequential_35513781:
??"
sequential_35513783:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_35513741sequential_35513743sequential_35513745sequential_35513747sequential_35513749sequential_35513751sequential_35513753sequential_35513755sequential_35513757sequential_35513759sequential_35513761sequential_35513763sequential_35513765sequential_35513767sequential_35513769sequential_35513771sequential_35513773sequential_35513775sequential_35513777sequential_35513779sequential_35513781sequential_35513783*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722$
"sequential/StatefulPartitionedCall?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_35513741sequential_35513743sequential_35513745sequential_35513747sequential_35513749sequential_35513751sequential_35513753sequential_35513755sequential_35513757sequential_35513759sequential_35513761sequential_35513763sequential_35513765sequential_35513767sequential_35513769sequential_35513771sequential_35513773sequential_35513775sequential_35513777sequential_35513779sequential_35513781sequential_35513783#^sequential/StatefulPartitionedCall*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722&
$sequential/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355136812
lambda/PartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513741*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513753*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513765*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513769*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513773*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513777* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513781* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/PartitionedCall:output:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
p
D__inference_lambda_layer_call_and_return_conditional_losses_35515248
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
__inference_loss_fn_0_35515808S
9conv1_1_kernel_regularizer_square_readvariableop_resource:		`
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv1_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
IdentityIdentity"conv1_1/kernel/Regularizer/mul:z:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp
?
?
E__inference_conv3_2_layer_call_and_return_conditional_losses_35512618

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
F
*__inference_dropout_layer_call_fn_35515690

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355126532
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_35516202
file_prefix'
assignvariableop_rmsprop_iter:	 *
 assignvariableop_1_rmsprop_decay: 2
(assignvariableop_2_rmsprop_learning_rate: -
#assignvariableop_3_rmsprop_momentum: (
assignvariableop_4_rmsprop_rho: ;
!assignvariableop_5_conv1_1_kernel:		`-
assignvariableop_6_conv1_1_bias:`:
,assignvariableop_7_batch_normalization_gamma:89
+assignvariableop_8_batch_normalization_beta:8@
2assignvariableop_9_batch_normalization_moving_mean:8E
7assignvariableop_10_batch_normalization_moving_variance:8=
"assignvariableop_11_conv2_1_kernel:`?/
 assignvariableop_12_conv2_1_bias:	?=
/assignvariableop_13_batch_normalization_1_gamma:<
.assignvariableop_14_batch_normalization_1_beta:C
5assignvariableop_15_batch_normalization_1_moving_mean:G
9assignvariableop_16_batch_normalization_1_moving_variance:>
"assignvariableop_17_conv3_1_kernel:??/
 assignvariableop_18_conv3_1_bias:	?>
"assignvariableop_19_conv3_2_kernel:??/
 assignvariableop_20_conv3_2_bias:	?>
"assignvariableop_21_conv3_3_kernel:??/
 assignvariableop_22_conv3_3_bias:	?4
 assignvariableop_23_dense_kernel:
??-
assignvariableop_24_dense_bias:	?6
"assignvariableop_25_dense_1_kernel:
??/
 assignvariableop_26_dense_1_bias:	?#
assignvariableop_27_total: #
assignvariableop_28_count: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: H
.assignvariableop_31_rmsprop_conv1_1_kernel_rms:		`:
,assignvariableop_32_rmsprop_conv1_1_bias_rms:`G
9assignvariableop_33_rmsprop_batch_normalization_gamma_rms:8F
8assignvariableop_34_rmsprop_batch_normalization_beta_rms:8I
.assignvariableop_35_rmsprop_conv2_1_kernel_rms:`?;
,assignvariableop_36_rmsprop_conv2_1_bias_rms:	?I
;assignvariableop_37_rmsprop_batch_normalization_1_gamma_rms:H
:assignvariableop_38_rmsprop_batch_normalization_1_beta_rms:J
.assignvariableop_39_rmsprop_conv3_1_kernel_rms:??;
,assignvariableop_40_rmsprop_conv3_1_bias_rms:	?J
.assignvariableop_41_rmsprop_conv3_2_kernel_rms:??;
,assignvariableop_42_rmsprop_conv3_2_bias_rms:	?J
.assignvariableop_43_rmsprop_conv3_3_kernel_rms:??;
,assignvariableop_44_rmsprop_conv3_3_bias_rms:	?@
,assignvariableop_45_rmsprop_dense_kernel_rms:
??9
*assignvariableop_46_rmsprop_dense_bias_rms:	?B
.assignvariableop_47_rmsprop_dense_1_kernel_rms:
??;
,assignvariableop_48_rmsprop_dense_1_bias_rms:	?
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_rmsprop_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_rmsprop_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_rmsprop_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_rmsprop_momentumIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_rhoIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv1_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv1_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp2assignvariableop_9_batch_normalization_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv2_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_1_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_1_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv3_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_conv3_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv3_2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp assignvariableop_20_conv3_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv3_3_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp assignvariableop_22_conv3_3_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_dense_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_dense_1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_rmsprop_conv1_1_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_rmsprop_conv1_1_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp9assignvariableop_33_rmsprop_batch_normalization_gamma_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_rmsprop_batch_normalization_beta_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_rmsprop_conv2_1_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_rmsprop_conv2_1_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_rmsprop_batch_normalization_1_gamma_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp:assignvariableop_38_rmsprop_batch_normalization_1_beta_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_rmsprop_conv3_1_kernel_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_rmsprop_conv3_1_bias_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp.assignvariableop_41_rmsprop_conv3_2_kernel_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_rmsprop_conv3_2_bias_rmsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_rmsprop_conv3_3_kernel_rmsIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_rmsprop_conv3_3_bias_rmsIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp,assignvariableop_45_rmsprop_dense_kernel_rmsIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_rmsprop_dense_bias_rmsIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp.assignvariableop_47_rmsprop_dense_1_kernel_rmsIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_rmsprop_dense_1_bias_rmsIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
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
?
U
)__inference_lambda_layer_call_fn_35515254
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355135672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_35512875

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
n
D__inference_lambda_layer_call_and_return_conditional_losses_35513681

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35512385

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515484

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_35512691

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35514982

inputs@
&conv1_1_conv2d_readvariableop_resource:		`5
'conv1_1_biasadd_readvariableop_resource:`9
+batch_normalization_readvariableop_resource:8;
-batch_normalization_readvariableop_1_resource:8J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:8L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:8A
&conv2_1_conv2d_readvariableop_resource:`?6
'conv2_1_biasadd_readvariableop_resource:	?;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:B
&conv3_1_conv2d_readvariableop_resource:??6
'conv3_1_biasadd_readvariableop_resource:	?B
&conv3_2_conv2d_readvariableop_resource:??6
'conv3_2_biasadd_readvariableop_resource:	?B
&conv3_3_conv2d_readvariableop_resource:??6
'conv3_3_biasadd_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/BiasAdd/ReadVariableOp?conv2_1/Conv2D/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/BiasAdd/ReadVariableOp?conv3_1/Conv2D/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/BiasAdd/ReadVariableOp?conv3_2/Conv2D/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/BiasAdd/ReadVariableOp?conv3_3/Conv2D/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dinputs%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
conv1_1/BiasAddx
conv1_1/ReluReluconv1_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
conv1_1/Relu?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv1_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2_1/Conv2D/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02
conv2_1/Conv2D/ReadVariableOp?
conv2_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0%conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2_1/Conv2D?
conv2_1/BiasAdd/ReadVariableOpReadVariableOp'conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv2_1/BiasAdd/ReadVariableOp?
conv2_1/BiasAddBiasAddconv2_1/Conv2D:output:0&conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2_1/BiasAddy
conv2_1/ReluReluconv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2_1/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv3_1/Conv2D/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_1/Conv2D/ReadVariableOp?
conv3_1/Conv2DConv2D max_pooling2d_1/MaxPool:output:0%conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv3_1/Conv2D?
conv3_1/BiasAdd/ReadVariableOpReadVariableOp'conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_1/BiasAdd/ReadVariableOp?
conv3_1/BiasAddBiasAddconv3_1/Conv2D:output:0&conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv3_1/BiasAddy
conv3_1/ReluReluconv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
conv3_1/Relu?
conv3_2/Conv2D/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_2/Conv2D/ReadVariableOp?
conv3_2/Conv2DConv2Dconv3_1/Relu:activations:0%conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3_2/Conv2D?
conv3_2/BiasAdd/ReadVariableOpReadVariableOp'conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_2/BiasAdd/ReadVariableOp?
conv3_2/BiasAddBiasAddconv3_2/Conv2D:output:0&conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_2/BiasAddy
conv3_2/ReluReluconv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_2/Relu?
conv3_3/Conv2D/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3_3/Conv2D/ReadVariableOp?
conv3_3/Conv2DConv2Dconv3_2/Relu:activations:0%conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3_3/Conv2D?
conv3_3/BiasAdd/ReadVariableOpReadVariableOp'conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_3/BiasAdd/ReadVariableOp?
conv3_3/BiasAddBiasAddconv3_3/Conv2D:output:0&conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_3/BiasAddy
conv3_3/ReluReluconv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_3/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
dropout/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout/Identityo
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Flatten/Const?
Flatten/ReshapeReshapedropout/Identity:output:0Flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
Flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulFlatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_1/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?

IdentityIdentitydense_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp1^conv1_1/kernel/Regularizer/Square/ReadVariableOp^conv2_1/BiasAdd/ReadVariableOp^conv2_1/Conv2D/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp^conv3_1/BiasAdd/ReadVariableOp^conv3_1/Conv2D/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp^conv3_2/BiasAdd/ReadVariableOp^conv3_2/Conv2D/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp^conv3_3/BiasAdd/ReadVariableOp^conv3_3/Conv2D/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2@
conv2_1/BiasAdd/ReadVariableOpconv2_1/BiasAdd/ReadVariableOp2>
conv2_1/Conv2D/ReadVariableOpconv2_1/Conv2D/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2@
conv3_1/BiasAdd/ReadVariableOpconv3_1/BiasAdd/ReadVariableOp2>
conv3_1/Conv2D/ReadVariableOpconv3_1/Conv2D/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2@
conv3_2/BiasAdd/ReadVariableOpconv3_2/BiasAdd/ReadVariableOp2>
conv3_2/Conv2D/ReadVariableOpconv3_2/Conv2D/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2@
conv3_3/BiasAdd/ReadVariableOpconv3_3/BiasAdd/ReadVariableOp2>
conv3_3/Conv2D/ReadVariableOpconv3_3/Conv2D/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
E__inference_conv3_1_layer_call_and_return_conditional_losses_35515595

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_35512836

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_1_layer_call_fn_35515760

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355126912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_35514810
inputs_0
inputs_1!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_355138532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?
?
6__inference_batch_normalization_layer_call_fn_35515416

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355129952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_35514282
input_1
input_2!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_355121812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515502

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_35515819T
9conv2_1_kernel_regularizer_square_readvariableop_resource:`?
identity??0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2_1_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
IdentityIdentity"conv2_1/kernel/Regularizer/mul:z:01^conv2_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp
?
?
(__inference_model_layer_call_fn_35514760
inputs_0
inputs_1!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_355136122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?
?
*__inference_conv2_1_layer_call_fn_35515448

inputs"
unknown:`?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2_1_layer_call_and_return_conditional_losses_355125442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????`: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
F
*__inference_Flatten_layer_call_fn_35515706

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Flatten_layer_call_and_return_conditional_losses_355126612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515346

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_35515738

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_355126802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_35515403

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355125162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_2_layer_call_fn_35512469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_355124632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515364

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_35513950
input_1
input_2!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_355138532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
?
(__inference_model_layer_call_fn_35513659
input_1
input_2!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_355136122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
?
C__inference_dense_layer_call_and_return_conditional_losses_35515729

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv3_2_layer_call_fn_35515636

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_2_layer_call_and_return_conditional_losses_355126182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_35512653

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2_1_layer_call_and_return_conditional_losses_35512544

inputs9
conv2d_readvariableop_resource:`?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
E__inference_conv3_2_layer_call_and_return_conditional_losses_35515627

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????		?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_35515788

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
U
)__inference_lambda_layer_call_fn_35515260
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355136812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35513373
conv1_1_input*
conv1_1_35513271:		`
conv1_1_35513273:`*
batch_normalization_35513276:8*
batch_normalization_35513278:8*
batch_normalization_35513280:8*
batch_normalization_35513282:8+
conv2_1_35513286:`?
conv2_1_35513288:	?,
batch_normalization_1_35513291:,
batch_normalization_1_35513293:,
batch_normalization_1_35513295:,
batch_normalization_1_35513297:,
conv3_1_35513301:??
conv3_1_35513303:	?,
conv3_2_35513306:??
conv3_2_35513308:	?,
conv3_3_35513311:??
conv3_3_35513313:	?"
dense_35513319:
??
dense_35513321:	?$
dense_1_35513325:
??
dense_1_35513327:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/StatefulPartitionedCall?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/StatefulPartitionedCall?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/StatefulPartitionedCall?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/StatefulPartitionedCall?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallconv1_1_inputconv1_1_35513271conv1_1_35513273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1_1_layer_call_and_return_conditional_losses_355124932!
conv1_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0batch_normalization_35513276batch_normalization_35513278batch_normalization_35513280batch_normalization_35513282*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355125162-
+batch_normalization/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_355123132
max_pooling2d/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_1_35513286conv2_1_35513288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2_1_layer_call_and_return_conditional_losses_355125442!
conv2_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0batch_normalization_1_35513291batch_normalization_1_35513293batch_normalization_1_35513295batch_normalization_1_35513297*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355125672/
-batch_normalization_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_355124512!
max_pooling2d_1/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_1_35513301conv3_1_35513303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_1_layer_call_and_return_conditional_losses_355125952!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_35513306conv3_2_35513308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_2_layer_call_and_return_conditional_losses_355126182!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_35513311conv3_3_35513313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_3_layer_call_and_return_conditional_losses_355126412!
conv3_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_355124632!
max_pooling2d_2/PartitionedCall?
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355126532
dropout/PartitionedCall?
Flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Flatten_layer_call_and_return_conditional_losses_355126612
Flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_35513319dense_35513321*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_355126802
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355126912
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_35513325dense_1_35513327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_355127102!
dense_1/StatefulPartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_1_35513271*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_1_35513286*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_1_35513301*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_2_35513306*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_3_35513311*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35513319* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_35513325* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall1^conv1_1/kernel/Regularizer/Square/ReadVariableOp ^conv2_1/StatefulPartitionedCall1^conv2_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_1/StatefulPartitionedCall1^conv3_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_2/StatefulPartitionedCall1^conv3_2/kernel/Regularizer/Square/ReadVariableOp ^conv3_3/StatefulPartitionedCall1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:^ Z
/
_output_shapes
:?????????@@
'
_user_specified_nameconv1_1_input
?
?
E__inference_dense_1_layer_call_and_return_conditional_losses_35512710

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv1_1_layer_call_and_return_conditional_losses_35512493

inputs8
conv2d_readvariableop_resource:		`-
biasadd_readvariableop_resource:`
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
Relu?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv1_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_35515546

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355123852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2_1_layer_call_and_return_conditional_losses_35515439

inputs9
conv2d_readvariableop_resource:`?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_1_layer_call_fn_35512457

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_355124512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_35512313

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_35515377

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+?????????8??????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355122032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_layer_call_and_return_conditional_losses_35515673

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515743

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?!
#__inference__wrapped_model_35512181
input_1
input_2Q
7model_sequential_conv1_1_conv2d_readvariableop_resource:		`F
8model_sequential_conv1_1_biasadd_readvariableop_resource:`J
<model_sequential_batch_normalization_readvariableop_resource:8L
>model_sequential_batch_normalization_readvariableop_1_resource:8[
Mmodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:8]
Omodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:8R
7model_sequential_conv2_1_conv2d_readvariableop_resource:`?G
8model_sequential_conv2_1_biasadd_readvariableop_resource:	?L
>model_sequential_batch_normalization_1_readvariableop_resource:N
@model_sequential_batch_normalization_1_readvariableop_1_resource:]
Omodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:_
Qmodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:S
7model_sequential_conv3_1_conv2d_readvariableop_resource:??G
8model_sequential_conv3_1_biasadd_readvariableop_resource:	?S
7model_sequential_conv3_2_conv2d_readvariableop_resource:??G
8model_sequential_conv3_2_biasadd_readvariableop_resource:	?S
7model_sequential_conv3_3_conv2d_readvariableop_resource:??G
8model_sequential_conv3_3_biasadd_readvariableop_resource:	?I
5model_sequential_dense_matmul_readvariableop_resource:
??E
6model_sequential_dense_biasadd_readvariableop_resource:	?K
7model_sequential_dense_1_matmul_readvariableop_resource:
??G
8model_sequential_dense_1_biasadd_readvariableop_resource:	?
identity??Dmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Fmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Fmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?Hmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?3model/sequential/batch_normalization/ReadVariableOp?5model/sequential/batch_normalization/ReadVariableOp_1?5model/sequential/batch_normalization/ReadVariableOp_2?5model/sequential/batch_normalization/ReadVariableOp_3?Fmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Hmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Hmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?Jmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?5model/sequential/batch_normalization_1/ReadVariableOp?7model/sequential/batch_normalization_1/ReadVariableOp_1?7model/sequential/batch_normalization_1/ReadVariableOp_2?7model/sequential/batch_normalization_1/ReadVariableOp_3?/model/sequential/conv1_1/BiasAdd/ReadVariableOp?1model/sequential/conv1_1/BiasAdd_1/ReadVariableOp?.model/sequential/conv1_1/Conv2D/ReadVariableOp?0model/sequential/conv1_1/Conv2D_1/ReadVariableOp?/model/sequential/conv2_1/BiasAdd/ReadVariableOp?1model/sequential/conv2_1/BiasAdd_1/ReadVariableOp?.model/sequential/conv2_1/Conv2D/ReadVariableOp?0model/sequential/conv2_1/Conv2D_1/ReadVariableOp?/model/sequential/conv3_1/BiasAdd/ReadVariableOp?1model/sequential/conv3_1/BiasAdd_1/ReadVariableOp?.model/sequential/conv3_1/Conv2D/ReadVariableOp?0model/sequential/conv3_1/Conv2D_1/ReadVariableOp?/model/sequential/conv3_2/BiasAdd/ReadVariableOp?1model/sequential/conv3_2/BiasAdd_1/ReadVariableOp?.model/sequential/conv3_2/Conv2D/ReadVariableOp?0model/sequential/conv3_2/Conv2D_1/ReadVariableOp?/model/sequential/conv3_3/BiasAdd/ReadVariableOp?1model/sequential/conv3_3/BiasAdd_1/ReadVariableOp?.model/sequential/conv3_3/Conv2D/ReadVariableOp?0model/sequential/conv3_3/Conv2D_1/ReadVariableOp?-model/sequential/dense/BiasAdd/ReadVariableOp?/model/sequential/dense/BiasAdd_1/ReadVariableOp?,model/sequential/dense/MatMul/ReadVariableOp?.model/sequential/dense/MatMul_1/ReadVariableOp?/model/sequential/dense_1/BiasAdd/ReadVariableOp?1model/sequential/dense_1/BiasAdd_1/ReadVariableOp?.model/sequential/dense_1/MatMul/ReadVariableOp?0model/sequential/dense_1/MatMul_1/ReadVariableOp?
.model/sequential/conv1_1/Conv2D/ReadVariableOpReadVariableOp7model_sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype020
.model/sequential/conv1_1/Conv2D/ReadVariableOp?
model/sequential/conv1_1/Conv2DConv2Dinput_16model/sequential/conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2!
model/sequential/conv1_1/Conv2D?
/model/sequential/conv1_1/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype021
/model/sequential/conv1_1/BiasAdd/ReadVariableOp?
 model/sequential/conv1_1/BiasAddBiasAdd(model/sequential/conv1_1/Conv2D:output:07model/sequential/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2"
 model/sequential/conv1_1/BiasAdd?
model/sequential/conv1_1/ReluRelu)model/sequential/conv1_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
model/sequential/conv1_1/Relu?
3model/sequential/batch_normalization/ReadVariableOpReadVariableOp<model_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype025
3model/sequential/batch_normalization/ReadVariableOp?
5model/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp>model_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5model/sequential/batch_normalization/ReadVariableOp_1?
Dmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02F
Dmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Fmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02H
Fmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5model/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3+model/sequential/conv1_1/Relu:activations:0;model/sequential/batch_normalization/ReadVariableOp:value:0=model/sequential/batch_normalization/ReadVariableOp_1:value:0Lmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 27
5model/sequential/batch_normalization/FusedBatchNormV3?
&model/sequential/max_pooling2d/MaxPoolMaxPool9model/sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2(
&model/sequential/max_pooling2d/MaxPool?
.model/sequential/conv2_1/Conv2D/ReadVariableOpReadVariableOp7model_sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype020
.model/sequential/conv2_1/Conv2D/ReadVariableOp?
model/sequential/conv2_1/Conv2DConv2D/model/sequential/max_pooling2d/MaxPool:output:06model/sequential/conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2!
model/sequential/conv2_1/Conv2D?
/model/sequential/conv2_1/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/conv2_1/BiasAdd/ReadVariableOp?
 model/sequential/conv2_1/BiasAddBiasAdd(model/sequential/conv2_1/Conv2D:output:07model/sequential/conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/sequential/conv2_1/BiasAdd?
model/sequential/conv2_1/ReluRelu)model/sequential/conv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/sequential/conv2_1/Relu?
5model/sequential/batch_normalization_1/ReadVariableOpReadVariableOp>model_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype027
5model/sequential/batch_normalization_1/ReadVariableOp?
7model/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp@model_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype029
7model/sequential/batch_normalization_1/ReadVariableOp_1?
Fmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02H
Fmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02J
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
7model/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+model/sequential/conv2_1/Relu:activations:0=model/sequential/batch_normalization_1/ReadVariableOp:value:0?model/sequential/batch_normalization_1/ReadVariableOp_1:value:0Nmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 29
7model/sequential/batch_normalization_1/FusedBatchNormV3?
(model/sequential/max_pooling2d_1/MaxPoolMaxPool;model/sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2*
(model/sequential/max_pooling2d_1/MaxPool?
.model/sequential/conv3_1/Conv2D/ReadVariableOpReadVariableOp7model_sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model/sequential/conv3_1/Conv2D/ReadVariableOp?
model/sequential/conv3_1/Conv2DConv2D1model/sequential/max_pooling2d_1/MaxPool:output:06model/sequential/conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2!
model/sequential/conv3_1/Conv2D?
/model/sequential/conv3_1/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/conv3_1/BiasAdd/ReadVariableOp?
 model/sequential/conv3_1/BiasAddBiasAdd(model/sequential/conv3_1/Conv2D:output:07model/sequential/conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2"
 model/sequential/conv3_1/BiasAdd?
model/sequential/conv3_1/ReluRelu)model/sequential/conv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
model/sequential/conv3_1/Relu?
.model/sequential/conv3_2/Conv2D/ReadVariableOpReadVariableOp7model_sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model/sequential/conv3_2/Conv2D/ReadVariableOp?
model/sequential/conv3_2/Conv2DConv2D+model/sequential/conv3_1/Relu:activations:06model/sequential/conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2!
model/sequential/conv3_2/Conv2D?
/model/sequential/conv3_2/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/conv3_2/BiasAdd/ReadVariableOp?
 model/sequential/conv3_2/BiasAddBiasAdd(model/sequential/conv3_2/Conv2D:output:07model/sequential/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/sequential/conv3_2/BiasAdd?
model/sequential/conv3_2/ReluRelu)model/sequential/conv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/sequential/conv3_2/Relu?
.model/sequential/conv3_3/Conv2D/ReadVariableOpReadVariableOp7model_sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model/sequential/conv3_3/Conv2D/ReadVariableOp?
model/sequential/conv3_3/Conv2DConv2D+model/sequential/conv3_2/Relu:activations:06model/sequential/conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2!
model/sequential/conv3_3/Conv2D?
/model/sequential/conv3_3/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/conv3_3/BiasAdd/ReadVariableOp?
 model/sequential/conv3_3/BiasAddBiasAdd(model/sequential/conv3_3/Conv2D:output:07model/sequential/conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/sequential/conv3_3/BiasAdd?
model/sequential/conv3_3/ReluRelu)model/sequential/conv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/sequential/conv3_3/Relu?
(model/sequential/max_pooling2d_2/MaxPoolMaxPool+model/sequential/conv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2*
(model/sequential/max_pooling2d_2/MaxPool?
!model/sequential/dropout/IdentityIdentity1model/sequential/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2#
!model/sequential/dropout/Identity?
model/sequential/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
model/sequential/Flatten/Const?
 model/sequential/Flatten/ReshapeReshape*model/sequential/dropout/Identity:output:0'model/sequential/Flatten/Const:output:0*
T0*(
_output_shapes
:??????????2"
 model/sequential/Flatten/Reshape?
,model/sequential/dense/MatMul/ReadVariableOpReadVariableOp5model_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,model/sequential/dense/MatMul/ReadVariableOp?
model/sequential/dense/MatMulMatMul)model/sequential/Flatten/Reshape:output:04model/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense/MatMul?
-model/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp6model_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model/sequential/dense/BiasAdd/ReadVariableOp?
model/sequential/dense/BiasAddBiasAdd'model/sequential/dense/MatMul:product:05model/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
model/sequential/dense/BiasAdd?
model/sequential/dense/ReluRelu'model/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense/Relu?
#model/sequential/dropout_1/IdentityIdentity)model/sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:??????????2%
#model/sequential/dropout_1/Identity?
.model/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7model_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.model/sequential/dense_1/MatMul/ReadVariableOp?
model/sequential/dense_1/MatMulMatMul,model/sequential/dropout_1/Identity:output:06model/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/sequential/dense_1/MatMul?
/model/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/dense_1/BiasAdd/ReadVariableOp?
 model/sequential/dense_1/BiasAddBiasAdd)model/sequential/dense_1/MatMul:product:07model/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model/sequential/dense_1/BiasAdd?
model/sequential/dense_1/ReluRelu)model/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense_1/Relu?
0model/sequential/conv1_1/Conv2D_1/ReadVariableOpReadVariableOp7model_sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0model/sequential/conv1_1/Conv2D_1/ReadVariableOp?
!model/sequential/conv1_1/Conv2D_1Conv2Dinput_28model/sequential/conv1_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2#
!model/sequential/conv1_1/Conv2D_1?
1model/sequential/conv1_1/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype023
1model/sequential/conv1_1/BiasAdd_1/ReadVariableOp?
"model/sequential/conv1_1/BiasAdd_1BiasAdd*model/sequential/conv1_1/Conv2D_1:output:09model/sequential/conv1_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2$
"model/sequential/conv1_1/BiasAdd_1?
model/sequential/conv1_1/Relu_1Relu+model/sequential/conv1_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????88`2!
model/sequential/conv1_1/Relu_1?
5model/sequential/batch_normalization/ReadVariableOp_2ReadVariableOp<model_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype027
5model/sequential/batch_normalization/ReadVariableOp_2?
5model/sequential/batch_normalization/ReadVariableOp_3ReadVariableOp>model_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5model/sequential/batch_normalization/ReadVariableOp_3?
Fmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpMmodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02H
Fmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?
Hmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpOmodel_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02J
Hmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?
7model/sequential/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3-model/sequential/conv1_1/Relu_1:activations:0=model/sequential/batch_normalization/ReadVariableOp_2:value:0=model/sequential/batch_normalization/ReadVariableOp_3:value:0Nmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Pmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 29
7model/sequential/batch_normalization/FusedBatchNormV3_1?
(model/sequential/max_pooling2d/MaxPool_1MaxPool;model/sequential/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2*
(model/sequential/max_pooling2d/MaxPool_1?
0model/sequential/conv2_1/Conv2D_1/ReadVariableOpReadVariableOp7model_sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0model/sequential/conv2_1/Conv2D_1/ReadVariableOp?
!model/sequential/conv2_1/Conv2D_1Conv2D1model/sequential/max_pooling2d/MaxPool_1:output:08model/sequential/conv2_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!model/sequential/conv2_1/Conv2D_1?
1model/sequential/conv2_1/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model/sequential/conv2_1/BiasAdd_1/ReadVariableOp?
"model/sequential/conv2_1/BiasAdd_1BiasAdd*model/sequential/conv2_1/Conv2D_1:output:09model/sequential/conv2_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model/sequential/conv2_1/BiasAdd_1?
model/sequential/conv2_1/Relu_1Relu+model/sequential/conv2_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2!
model/sequential/conv2_1/Relu_1?
7model/sequential/batch_normalization_1/ReadVariableOp_2ReadVariableOp>model_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype029
7model/sequential/batch_normalization_1/ReadVariableOp_2?
7model/sequential/batch_normalization_1/ReadVariableOp_3ReadVariableOp@model_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype029
7model/sequential/batch_normalization_1/ReadVariableOp_3?
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOmodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02J
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?
Jmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQmodel_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?
9model/sequential/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3-model/sequential/conv2_1/Relu_1:activations:0?model/sequential/batch_normalization_1/ReadVariableOp_2:value:0?model/sequential/batch_normalization_1/ReadVariableOp_3:value:0Pmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Rmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2;
9model/sequential/batch_normalization_1/FusedBatchNormV3_1?
*model/sequential/max_pooling2d_1/MaxPool_1MaxPool=model/sequential/batch_normalization_1/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2,
*model/sequential/max_pooling2d_1/MaxPool_1?
0model/sequential/conv3_1/Conv2D_1/ReadVariableOpReadVariableOp7model_sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model/sequential/conv3_1/Conv2D_1/ReadVariableOp?
!model/sequential/conv3_1/Conv2D_1Conv2D3model/sequential/max_pooling2d_1/MaxPool_1:output:08model/sequential/conv3_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2#
!model/sequential/conv3_1/Conv2D_1?
1model/sequential/conv3_1/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model/sequential/conv3_1/BiasAdd_1/ReadVariableOp?
"model/sequential/conv3_1/BiasAdd_1BiasAdd*model/sequential/conv3_1/Conv2D_1:output:09model/sequential/conv3_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2$
"model/sequential/conv3_1/BiasAdd_1?
model/sequential/conv3_1/Relu_1Relu+model/sequential/conv3_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????		?2!
model/sequential/conv3_1/Relu_1?
0model/sequential/conv3_2/Conv2D_1/ReadVariableOpReadVariableOp7model_sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model/sequential/conv3_2/Conv2D_1/ReadVariableOp?
!model/sequential/conv3_2/Conv2D_1Conv2D-model/sequential/conv3_1/Relu_1:activations:08model/sequential/conv3_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!model/sequential/conv3_2/Conv2D_1?
1model/sequential/conv3_2/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model/sequential/conv3_2/BiasAdd_1/ReadVariableOp?
"model/sequential/conv3_2/BiasAdd_1BiasAdd*model/sequential/conv3_2/Conv2D_1:output:09model/sequential/conv3_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model/sequential/conv3_2/BiasAdd_1?
model/sequential/conv3_2/Relu_1Relu+model/sequential/conv3_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2!
model/sequential/conv3_2/Relu_1?
0model/sequential/conv3_3/Conv2D_1/ReadVariableOpReadVariableOp7model_sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model/sequential/conv3_3/Conv2D_1/ReadVariableOp?
!model/sequential/conv3_3/Conv2D_1Conv2D-model/sequential/conv3_2/Relu_1:activations:08model/sequential/conv3_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!model/sequential/conv3_3/Conv2D_1?
1model/sequential/conv3_3/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model/sequential/conv3_3/BiasAdd_1/ReadVariableOp?
"model/sequential/conv3_3/BiasAdd_1BiasAdd*model/sequential/conv3_3/Conv2D_1:output:09model/sequential/conv3_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"model/sequential/conv3_3/BiasAdd_1?
model/sequential/conv3_3/Relu_1Relu+model/sequential/conv3_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2!
model/sequential/conv3_3/Relu_1?
*model/sequential/max_pooling2d_2/MaxPool_1MaxPool-model/sequential/conv3_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2,
*model/sequential/max_pooling2d_2/MaxPool_1?
#model/sequential/dropout/Identity_1Identity3model/sequential/max_pooling2d_2/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????2%
#model/sequential/dropout/Identity_1?
 model/sequential/Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????   2"
 model/sequential/Flatten/Const_1?
"model/sequential/Flatten/Reshape_1Reshape,model/sequential/dropout/Identity_1:output:0)model/sequential/Flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????2$
"model/sequential/Flatten/Reshape_1?
.model/sequential/dense/MatMul_1/ReadVariableOpReadVariableOp5model_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.model/sequential/dense/MatMul_1/ReadVariableOp?
model/sequential/dense/MatMul_1MatMul+model/sequential/Flatten/Reshape_1:output:06model/sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/sequential/dense/MatMul_1?
/model/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp6model_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/sequential/dense/BiasAdd_1/ReadVariableOp?
 model/sequential/dense/BiasAdd_1BiasAdd)model/sequential/dense/MatMul_1:product:07model/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model/sequential/dense/BiasAdd_1?
model/sequential/dense/Relu_1Relu)model/sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense/Relu_1?
%model/sequential/dropout_1/Identity_1Identity+model/sequential/dense/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2'
%model/sequential/dropout_1/Identity_1?
0model/sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOp7model_sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0model/sequential/dense_1/MatMul_1/ReadVariableOp?
!model/sequential/dense_1/MatMul_1MatMul.model/sequential/dropout_1/Identity_1:output:08model/sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model/sequential/dense_1/MatMul_1?
1model/sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp8model_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model/sequential/dense_1/BiasAdd_1/ReadVariableOp?
"model/sequential/dense_1/BiasAdd_1BiasAdd+model/sequential/dense_1/MatMul_1:product:09model/sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model/sequential/dense_1/BiasAdd_1?
model/sequential/dense_1/Relu_1Relu+model/sequential/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2!
model/sequential/dense_1/Relu_1?
model/lambda/subSub+model/sequential/dense_1/Relu:activations:0-model/sequential/dense_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
model/lambda/sub}
model/lambda/SquareSquaremodel/lambda/sub:z:0*
T0*(
_output_shapes
:??????????2
model/lambda/Square?
"model/lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/lambda/Sum/reduction_indices?
model/lambda/SumSummodel/lambda/Square:y:0+model/lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
model/lambda/Summ
model/lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lambda/Const?
model/lambda/MaximumMaximummodel/lambda/Sum:output:0model/lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
model/lambda/Maximumz
model/lambda/SqrtSqrtmodel/lambda/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/lambda/Sqrt?
IdentityIdentitymodel/lambda/Sqrt:y:0E^model/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpG^model/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1G^model/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpI^model/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_14^model/sequential/batch_normalization/ReadVariableOp6^model/sequential/batch_normalization/ReadVariableOp_16^model/sequential/batch_normalization/ReadVariableOp_26^model/sequential/batch_normalization/ReadVariableOp_3G^model/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^model/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1I^model/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpK^model/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_16^model/sequential/batch_normalization_1/ReadVariableOp8^model/sequential/batch_normalization_1/ReadVariableOp_18^model/sequential/batch_normalization_1/ReadVariableOp_28^model/sequential/batch_normalization_1/ReadVariableOp_30^model/sequential/conv1_1/BiasAdd/ReadVariableOp2^model/sequential/conv1_1/BiasAdd_1/ReadVariableOp/^model/sequential/conv1_1/Conv2D/ReadVariableOp1^model/sequential/conv1_1/Conv2D_1/ReadVariableOp0^model/sequential/conv2_1/BiasAdd/ReadVariableOp2^model/sequential/conv2_1/BiasAdd_1/ReadVariableOp/^model/sequential/conv2_1/Conv2D/ReadVariableOp1^model/sequential/conv2_1/Conv2D_1/ReadVariableOp0^model/sequential/conv3_1/BiasAdd/ReadVariableOp2^model/sequential/conv3_1/BiasAdd_1/ReadVariableOp/^model/sequential/conv3_1/Conv2D/ReadVariableOp1^model/sequential/conv3_1/Conv2D_1/ReadVariableOp0^model/sequential/conv3_2/BiasAdd/ReadVariableOp2^model/sequential/conv3_2/BiasAdd_1/ReadVariableOp/^model/sequential/conv3_2/Conv2D/ReadVariableOp1^model/sequential/conv3_2/Conv2D_1/ReadVariableOp0^model/sequential/conv3_3/BiasAdd/ReadVariableOp2^model/sequential/conv3_3/BiasAdd_1/ReadVariableOp/^model/sequential/conv3_3/Conv2D/ReadVariableOp1^model/sequential/conv3_3/Conv2D_1/ReadVariableOp.^model/sequential/dense/BiasAdd/ReadVariableOp0^model/sequential/dense/BiasAdd_1/ReadVariableOp-^model/sequential/dense/MatMul/ReadVariableOp/^model/sequential/dense/MatMul_1/ReadVariableOp0^model/sequential/dense_1/BiasAdd/ReadVariableOp2^model/sequential/dense_1/BiasAdd_1/ReadVariableOp/^model/sequential/dense_1/MatMul/ReadVariableOp1^model/sequential/dense_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2?
Dmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Fmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodel/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Fmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpFmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2?
Hmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Hmodel/sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12j
3model/sequential/batch_normalization/ReadVariableOp3model/sequential/batch_normalization/ReadVariableOp2n
5model/sequential/batch_normalization/ReadVariableOp_15model/sequential/batch_normalization/ReadVariableOp_12n
5model/sequential/batch_normalization/ReadVariableOp_25model/sequential/batch_normalization/ReadVariableOp_22n
5model/sequential/batch_normalization/ReadVariableOp_35model/sequential/batch_normalization/ReadVariableOp_32?
Fmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodel/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Hmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpHmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
Jmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Jmodel/sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12n
5model/sequential/batch_normalization_1/ReadVariableOp5model/sequential/batch_normalization_1/ReadVariableOp2r
7model/sequential/batch_normalization_1/ReadVariableOp_17model/sequential/batch_normalization_1/ReadVariableOp_12r
7model/sequential/batch_normalization_1/ReadVariableOp_27model/sequential/batch_normalization_1/ReadVariableOp_22r
7model/sequential/batch_normalization_1/ReadVariableOp_37model/sequential/batch_normalization_1/ReadVariableOp_32b
/model/sequential/conv1_1/BiasAdd/ReadVariableOp/model/sequential/conv1_1/BiasAdd/ReadVariableOp2f
1model/sequential/conv1_1/BiasAdd_1/ReadVariableOp1model/sequential/conv1_1/BiasAdd_1/ReadVariableOp2`
.model/sequential/conv1_1/Conv2D/ReadVariableOp.model/sequential/conv1_1/Conv2D/ReadVariableOp2d
0model/sequential/conv1_1/Conv2D_1/ReadVariableOp0model/sequential/conv1_1/Conv2D_1/ReadVariableOp2b
/model/sequential/conv2_1/BiasAdd/ReadVariableOp/model/sequential/conv2_1/BiasAdd/ReadVariableOp2f
1model/sequential/conv2_1/BiasAdd_1/ReadVariableOp1model/sequential/conv2_1/BiasAdd_1/ReadVariableOp2`
.model/sequential/conv2_1/Conv2D/ReadVariableOp.model/sequential/conv2_1/Conv2D/ReadVariableOp2d
0model/sequential/conv2_1/Conv2D_1/ReadVariableOp0model/sequential/conv2_1/Conv2D_1/ReadVariableOp2b
/model/sequential/conv3_1/BiasAdd/ReadVariableOp/model/sequential/conv3_1/BiasAdd/ReadVariableOp2f
1model/sequential/conv3_1/BiasAdd_1/ReadVariableOp1model/sequential/conv3_1/BiasAdd_1/ReadVariableOp2`
.model/sequential/conv3_1/Conv2D/ReadVariableOp.model/sequential/conv3_1/Conv2D/ReadVariableOp2d
0model/sequential/conv3_1/Conv2D_1/ReadVariableOp0model/sequential/conv3_1/Conv2D_1/ReadVariableOp2b
/model/sequential/conv3_2/BiasAdd/ReadVariableOp/model/sequential/conv3_2/BiasAdd/ReadVariableOp2f
1model/sequential/conv3_2/BiasAdd_1/ReadVariableOp1model/sequential/conv3_2/BiasAdd_1/ReadVariableOp2`
.model/sequential/conv3_2/Conv2D/ReadVariableOp.model/sequential/conv3_2/Conv2D/ReadVariableOp2d
0model/sequential/conv3_2/Conv2D_1/ReadVariableOp0model/sequential/conv3_2/Conv2D_1/ReadVariableOp2b
/model/sequential/conv3_3/BiasAdd/ReadVariableOp/model/sequential/conv3_3/BiasAdd/ReadVariableOp2f
1model/sequential/conv3_3/BiasAdd_1/ReadVariableOp1model/sequential/conv3_3/BiasAdd_1/ReadVariableOp2`
.model/sequential/conv3_3/Conv2D/ReadVariableOp.model/sequential/conv3_3/Conv2D/ReadVariableOp2d
0model/sequential/conv3_3/Conv2D_1/ReadVariableOp0model/sequential/conv3_3/Conv2D_1/ReadVariableOp2^
-model/sequential/dense/BiasAdd/ReadVariableOp-model/sequential/dense/BiasAdd/ReadVariableOp2b
/model/sequential/dense/BiasAdd_1/ReadVariableOp/model/sequential/dense/BiasAdd_1/ReadVariableOp2\
,model/sequential/dense/MatMul/ReadVariableOp,model/sequential/dense/MatMul/ReadVariableOp2`
.model/sequential/dense/MatMul_1/ReadVariableOp.model/sequential/dense/MatMul_1/ReadVariableOp2b
/model/sequential/dense_1/BiasAdd/ReadVariableOp/model/sequential/dense_1/BiasAdd/ReadVariableOp2f
1model/sequential/dense_1/BiasAdd_1/ReadVariableOp1model/sequential/dense_1/BiasAdd_1/ReadVariableOp2`
.model/sequential/dense_1/MatMul/ReadVariableOp.model/sequential/dense_1/MatMul/ReadVariableOp2d
0model/sequential/dense_1/MatMul_1/ReadVariableOp0model/sequential/dense_1/MatMul_1/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
?
__inference_loss_fn_6_35515874M
9dense_1_kernel_regularizer_square_readvariableop_resource:
??
identity??0dense_1/kernel/Regularizer/Square/ReadVariableOp?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
?
L
0__inference_max_pooling2d_layer_call_fn_35512319

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_355123132
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_layer_call_and_return_conditional_losses_35512680

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35512341

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_35515390

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+?????????8??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355122472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_35515841U
9conv3_2_kernel_regularizer_square_readvariableop_resource:??
identity??0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv3_2_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
IdentityIdentity"conv3_2/kernel/Regularizer/mul:z:01^conv3_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_35513478
conv1_1_input*
conv1_1_35513376:		`
conv1_1_35513378:`*
batch_normalization_35513381:8*
batch_normalization_35513383:8*
batch_normalization_35513385:8*
batch_normalization_35513387:8+
conv2_1_35513391:`?
conv2_1_35513393:	?,
batch_normalization_1_35513396:,
batch_normalization_1_35513398:,
batch_normalization_1_35513400:,
batch_normalization_1_35513402:,
conv3_1_35513406:??
conv3_1_35513408:	?,
conv3_2_35513411:??
conv3_2_35513413:	?,
conv3_3_35513416:??
conv3_3_35513418:	?"
dense_35513424:
??
dense_35513426:	?$
dense_1_35513430:
??
dense_1_35513432:	?
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?conv2_1/StatefulPartitionedCall?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?conv3_1/StatefulPartitionedCall?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?conv3_2/StatefulPartitionedCall?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?conv3_3/StatefulPartitionedCall?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?.dense/kernel/Regularizer/Square/ReadVariableOp?dense_1/StatefulPartitionedCall?0dense_1/kernel/Regularizer/Square/ReadVariableOp?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCallconv1_1_inputconv1_1_35513376conv1_1_35513378*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1_1_layer_call_and_return_conditional_losses_355124932!
conv1_1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(conv1_1/StatefulPartitionedCall:output:0batch_normalization_35513381batch_normalization_35513383batch_normalization_35513385batch_normalization_35513387*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_355129952-
+batch_normalization/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_355123132
max_pooling2d/PartitionedCall?
conv2_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_1_35513391conv2_1_35513393*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2_1_layer_call_and_return_conditional_losses_355125442!
conv2_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(conv2_1/StatefulPartitionedCall:output:0batch_normalization_1_35513396batch_normalization_1_35513398batch_normalization_1_35513400batch_normalization_1_35513402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355129412/
-batch_normalization_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_355124512!
max_pooling2d_1/PartitionedCall?
conv3_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv3_1_35513406conv3_1_35513408*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_1_layer_call_and_return_conditional_losses_355125952!
conv3_1/StatefulPartitionedCall?
conv3_2/StatefulPartitionedCallStatefulPartitionedCall(conv3_1/StatefulPartitionedCall:output:0conv3_2_35513411conv3_2_35513413*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_2_layer_call_and_return_conditional_losses_355126182!
conv3_2/StatefulPartitionedCall?
conv3_3/StatefulPartitionedCallStatefulPartitionedCall(conv3_2/StatefulPartitionedCall:output:0conv3_3_35513416conv3_3_35513418*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_3_layer_call_and_return_conditional_losses_355126412!
conv3_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(conv3_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_355124632!
max_pooling2d_2/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355128752!
dropout/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_Flatten_layer_call_and_return_conditional_losses_355126612
Flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_35513424dense_35513426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_355126802
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_355128362#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_35513430dense_1_35513432*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_355127102!
dense_1/StatefulPartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_1_35513376*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_1_35513391*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_1_35513406*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_2_35513411*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_3_35513416*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35513424* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_35513430* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall1^conv1_1/kernel/Regularizer/Square/ReadVariableOp ^conv2_1/StatefulPartitionedCall1^conv2_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_1/StatefulPartitionedCall1^conv3_1/kernel/Regularizer/Square/ReadVariableOp ^conv3_2/StatefulPartitionedCall1^conv3_2/kernel/Regularizer/Square/ReadVariableOp ^conv3_3/StatefulPartitionedCall1^conv3_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2_1/StatefulPartitionedCallconv2_1/StatefulPartitionedCall2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_1/StatefulPartitionedCallconv3_1/StatefulPartitionedCall2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3_2/StatefulPartitionedCallconv3_2/StatefulPartitionedCall2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3_3/StatefulPartitionedCallconv3_3/StatefulPartitionedCall2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????@@
'
_user_specified_nameconv1_1_input
?
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515755

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?c
?

C__inference_model_layer_call_and_return_conditional_losses_35514066
input_1
input_2-
sequential_35513954:		`!
sequential_35513956:`!
sequential_35513958:8!
sequential_35513960:8!
sequential_35513962:8!
sequential_35513964:8.
sequential_35513966:`?"
sequential_35513968:	?!
sequential_35513970:!
sequential_35513972:!
sequential_35513974:!
sequential_35513976:/
sequential_35513978:??"
sequential_35513980:	?/
sequential_35513982:??"
sequential_35513984:	?/
sequential_35513986:??"
sequential_35513988:	?'
sequential_35513990:
??"
sequential_35513992:	?'
sequential_35513994:
??"
sequential_35513996:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_35513954sequential_35513956sequential_35513958sequential_35513960sequential_35513962sequential_35513964sequential_35513966sequential_35513968sequential_35513970sequential_35513972sequential_35513974sequential_35513976sequential_35513978sequential_35513980sequential_35513982sequential_35513984sequential_35513986sequential_35513988sequential_35513990sequential_35513992sequential_35513994sequential_35513996*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592$
"sequential/StatefulPartitionedCall?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_35513954sequential_35513956sequential_35513958sequential_35513960sequential_35513962sequential_35513964sequential_35513966sequential_35513968sequential_35513970sequential_35513972sequential_35513974sequential_35513976sequential_35513978sequential_35513980sequential_35513982sequential_35513984sequential_35513986sequential_35513988sequential_35513990sequential_35513992sequential_35513994sequential_35513996*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592&
$sequential/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355135672
lambda/PartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513954*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513966*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513978*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513982*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513986*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513990* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513994* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/PartitionedCall:output:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????@@
!
_user_specified_name	input_2
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515328

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+?????????8??????????????????:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_layer_call_and_return_conditional_losses_35515685

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_35515797

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_355127102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
p
D__inference_lambda_layer_call_and_return_conditional_losses_35515236
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
SumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constm
MaximumMaximumSum:output:0Const:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35512567

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_35515572

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355129412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_35515224

inputs!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_35515175

inputs!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_sequential_layer_call_fn_35513268
conv1_1_input!
unknown:		`
	unknown_0:`
	unknown_1:8
	unknown_2:8
	unknown_3:8
	unknown_4:8$
	unknown_5:`?
	unknown_6:	?
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:
??

unknown_18:	?

unknown_19:
??

unknown_20:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355131722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????@@
'
_user_specified_nameconv1_1_input
?a
?
!__inference__traced_save_35516045
file_prefix+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_conv2_1_kernel_read_readvariableop+
'savev2_conv2_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_conv3_1_kernel_read_readvariableop+
'savev2_conv3_1_bias_read_readvariableop-
)savev2_conv3_2_kernel_read_readvariableop+
'savev2_conv3_2_bias_read_readvariableop-
)savev2_conv3_3_kernel_read_readvariableop+
'savev2_conv3_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_rmsprop_conv1_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv1_1_bias_rms_read_readvariableopD
@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableopC
?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop9
5savev2_rmsprop_conv2_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv2_1_bias_rms_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_1_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_1_beta_rms_read_readvariableop9
5savev2_rmsprop_conv3_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv3_1_bias_rms_read_readvariableop9
5savev2_rmsprop_conv3_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv3_2_bias_rms_read_readvariableop9
5savev2_rmsprop_conv3_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv3_3_bias_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_conv2_1_kernel_read_readvariableop'savev2_conv2_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_conv3_1_kernel_read_readvariableop'savev2_conv3_1_bias_read_readvariableop)savev2_conv3_2_kernel_read_readvariableop'savev2_conv3_2_bias_read_readvariableop)savev2_conv3_3_kernel_read_readvariableop'savev2_conv3_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_rmsprop_conv1_1_kernel_rms_read_readvariableop3savev2_rmsprop_conv1_1_bias_rms_read_readvariableop@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableop?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop5savev2_rmsprop_conv2_1_kernel_rms_read_readvariableop3savev2_rmsprop_conv2_1_bias_rms_read_readvariableopBsavev2_rmsprop_batch_normalization_1_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_1_beta_rms_read_readvariableop5savev2_rmsprop_conv3_1_kernel_rms_read_readvariableop3savev2_rmsprop_conv3_1_bias_rms_read_readvariableop5savev2_rmsprop_conv3_2_kernel_rms_read_readvariableop3savev2_rmsprop_conv3_2_bias_rms_read_readvariableop5savev2_rmsprop_conv3_3_kernel_rms_read_readvariableop3savev2_rmsprop_conv3_3_bias_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :		`:`:8:8:8:8:`?:?:::::??:?:??:?:??:?:
??:?:
??:?: : : : :		`:`:8:8:`?:?:::??:?:??:?:??:?:
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:		`: 

_output_shapes
:`: 

_output_shapes
:8: 	

_output_shapes
:8: 


_output_shapes
:8: 

_output_shapes
:8:-)
'
_output_shapes
:`?:!

_output_shapes	
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:		`: !

_output_shapes
:`: "

_output_shapes
:8: #

_output_shapes
:8:-$)
'
_output_shapes
:`?:!%

_output_shapes	
:?: &

_output_shapes
:: '

_output_shapes
::.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:.**
(
_output_shapes
:??:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:2

_output_shapes
: 
ȯ
?$
C__inference_model_layer_call_and_return_conditional_losses_35514710
inputs_0
inputs_1K
1sequential_conv1_1_conv2d_readvariableop_resource:		`@
2sequential_conv1_1_biasadd_readvariableop_resource:`D
6sequential_batch_normalization_readvariableop_resource:8F
8sequential_batch_normalization_readvariableop_1_resource:8U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:8W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:8L
1sequential_conv2_1_conv2d_readvariableop_resource:`?A
2sequential_conv2_1_biasadd_readvariableop_resource:	?F
8sequential_batch_normalization_1_readvariableop_resource:H
:sequential_batch_normalization_1_readvariableop_1_resource:W
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:Y
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:M
1sequential_conv3_1_conv2d_readvariableop_resource:??A
2sequential_conv3_1_biasadd_readvariableop_resource:	?M
1sequential_conv3_2_conv2d_readvariableop_resource:??A
2sequential_conv3_2_biasadd_readvariableop_resource:	?M
1sequential_conv3_3_conv2d_readvariableop_resource:??A
2sequential_conv3_3_biasadd_readvariableop_resource:	?C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?E
1sequential_dense_1_matmul_readvariableop_resource:
??A
2sequential_dense_1_biasadd_readvariableop_resource:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?/sequential/batch_normalization/AssignNewValue_2?/sequential/batch_normalization/AssignNewValue_3?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?/sequential/batch_normalization/ReadVariableOp_2?/sequential/batch_normalization/ReadVariableOp_3?/sequential/batch_normalization_1/AssignNewValue?1sequential/batch_normalization_1/AssignNewValue_1?1sequential/batch_normalization_1/AssignNewValue_2?1sequential/batch_normalization_1/AssignNewValue_3?@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?/sequential/batch_normalization_1/ReadVariableOp?1sequential/batch_normalization_1/ReadVariableOp_1?1sequential/batch_normalization_1/ReadVariableOp_2?1sequential/batch_normalization_1/ReadVariableOp_3?)sequential/conv1_1/BiasAdd/ReadVariableOp?+sequential/conv1_1/BiasAdd_1/ReadVariableOp?(sequential/conv1_1/Conv2D/ReadVariableOp?*sequential/conv1_1/Conv2D_1/ReadVariableOp?)sequential/conv2_1/BiasAdd/ReadVariableOp?+sequential/conv2_1/BiasAdd_1/ReadVariableOp?(sequential/conv2_1/Conv2D/ReadVariableOp?*sequential/conv2_1/Conv2D_1/ReadVariableOp?)sequential/conv3_1/BiasAdd/ReadVariableOp?+sequential/conv3_1/BiasAdd_1/ReadVariableOp?(sequential/conv3_1/Conv2D/ReadVariableOp?*sequential/conv3_1/Conv2D_1/ReadVariableOp?)sequential/conv3_2/BiasAdd/ReadVariableOp?+sequential/conv3_2/BiasAdd_1/ReadVariableOp?(sequential/conv3_2/Conv2D/ReadVariableOp?*sequential/conv3_2/Conv2D_1/ReadVariableOp?)sequential/conv3_3/BiasAdd/ReadVariableOp?+sequential/conv3_3/BiasAdd_1/ReadVariableOp?(sequential/conv3_3/Conv2D/ReadVariableOp?*sequential/conv3_3/Conv2D_1/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/BiasAdd_1/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?(sequential/dense/MatMul_1/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/BiasAdd_1/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?*sequential/dense_1/MatMul_1/ReadVariableOp?
(sequential/conv1_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02*
(sequential/conv1_1/Conv2D/ReadVariableOp?
sequential/conv1_1/Conv2DConv2Dinputs_00sequential/conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
sequential/conv1_1/Conv2D?
)sequential/conv1_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02+
)sequential/conv1_1/BiasAdd/ReadVariableOp?
sequential/conv1_1/BiasAddBiasAdd"sequential/conv1_1/Conv2D:output:01sequential/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/BiasAdd?
sequential/conv1_1/ReluRelu#sequential/conv1_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/Relu?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3%sequential/conv1_1/Relu:activations:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
(sequential/conv2_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02*
(sequential/conv2_1/Conv2D/ReadVariableOp?
sequential/conv2_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:00sequential/conv2_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv2_1/Conv2D?
)sequential/conv2_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv2_1/BiasAdd/ReadVariableOp?
sequential/conv2_1/BiasAddBiasAdd"sequential/conv2_1/Conv2D:output:01sequential/conv2_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/BiasAdd?
sequential/conv2_1/ReluRelu#sequential/conv2_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/Relu?
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization_1/ReadVariableOp?
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%sequential/conv2_1/Relu:activations:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=23
1sequential/batch_normalization_1/FusedBatchNormV3?
/sequential/batch_normalization_1/AssignNewValueAssignVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype021
/sequential/batch_normalization_1/AssignNewValue?
1sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0C^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype023
1sequential/batch_normalization_1/AssignNewValue_1?
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
(sequential/conv3_1/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_1/Conv2D/ReadVariableOp?
sequential/conv3_1/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:00sequential/conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
sequential/conv3_1/Conv2D?
)sequential/conv3_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_1/BiasAdd/ReadVariableOp?
sequential/conv3_1/BiasAddBiasAdd"sequential/conv3_1/Conv2D:output:01sequential/conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/BiasAdd?
sequential/conv3_1/ReluRelu#sequential/conv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/Relu?
(sequential/conv3_2/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_2/Conv2D/ReadVariableOp?
sequential/conv3_2/Conv2DConv2D%sequential/conv3_1/Relu:activations:00sequential/conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_2/Conv2D?
)sequential/conv3_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_2/BiasAdd/ReadVariableOp?
sequential/conv3_2/BiasAddBiasAdd"sequential/conv3_2/Conv2D:output:01sequential/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/BiasAdd?
sequential/conv3_2/ReluRelu#sequential/conv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/Relu?
(sequential/conv3_3/Conv2D/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential/conv3_3/Conv2D/ReadVariableOp?
sequential/conv3_3/Conv2DConv2D%sequential/conv3_2/Relu:activations:00sequential/conv3_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_3/Conv2D?
)sequential/conv3_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/conv3_3/BiasAdd/ReadVariableOp?
sequential/conv3_3/BiasAddBiasAdd"sequential/conv3_3/Conv2D:output:01sequential/conv3_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/BiasAdd?
sequential/conv3_3/ReluRelu#sequential/conv3_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/Relu?
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/conv3_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool?
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2"
 sequential/dropout/dropout/Const?
sequential/dropout/dropout/MulMul+sequential/max_pooling2d_2/MaxPool:output:0)sequential/dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2 
sequential/dropout/dropout/Mul?
 sequential/dropout/dropout/ShapeShape+sequential/max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2"
 sequential/dropout/dropout/Shape?
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype029
7sequential/dropout/dropout/random_uniform/RandomUniform?
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2+
)sequential/dropout/dropout/GreaterEqual/y?
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2)
'sequential/dropout/dropout/GreaterEqual?
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2!
sequential/dropout/dropout/Cast?
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2"
 sequential/dropout/dropout/Mul_1?
sequential/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/Flatten/Const?
sequential/Flatten/ReshapeReshape$sequential/dropout/dropout/Mul_1:z:0!sequential/Flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/Flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/Flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"sequential/dropout_1/dropout/Const?
 sequential/dropout_1/dropout/MulMul#sequential/dense/Relu:activations:0+sequential/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential/dropout_1/dropout/Mul?
"sequential/dropout_1/dropout/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:2$
"sequential/dropout_1/dropout/Shape?
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02;
9sequential/dropout_1/dropout/random_uniform/RandomUniform?
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+sequential/dropout_1/dropout/GreaterEqual/y?
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)sequential/dropout_1/dropout/GreaterEqual?
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!sequential/dropout_1/dropout/Cast?
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"sequential/dropout_1/dropout/Mul_1?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul&sequential/dropout_1/dropout/Mul_1:z:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
*sequential/conv1_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02,
*sequential/conv1_1/Conv2D_1/ReadVariableOp?
sequential/conv1_1/Conv2D_1Conv2Dinputs_12sequential/conv1_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
sequential/conv1_1/Conv2D_1?
+sequential/conv1_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02-
+sequential/conv1_1/BiasAdd_1/ReadVariableOp?
sequential/conv1_1/BiasAdd_1BiasAdd$sequential/conv1_1/Conv2D_1:output:03sequential/conv1_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/BiasAdd_1?
sequential/conv1_1/Relu_1Relu%sequential/conv1_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????88`2
sequential/conv1_1/Relu_1?
/sequential/batch_normalization/ReadVariableOp_2ReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_2?
/sequential/batch_normalization/ReadVariableOp_3ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_3?
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource.^sequential/batch_normalization/AssignNewValue*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource0^sequential/batch_normalization/AssignNewValue_1*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?
1sequential/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3'sequential/conv1_1/Relu_1:activations:07sequential/batch_normalization/ReadVariableOp_2:value:07sequential/batch_normalization/ReadVariableOp_3:value:0Hsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Jsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=23
1sequential/batch_normalization/FusedBatchNormV3_1?
/sequential/batch_normalization/AssignNewValue_2AssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization/FusedBatchNormV3_1:batch_mean:0.^sequential/batch_normalization/AssignNewValueA^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_2?
/sequential/batch_normalization/AssignNewValue_3AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization/FusedBatchNormV3_1:batch_variance:00^sequential/batch_normalization/AssignNewValue_1C^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_3?
"sequential/max_pooling2d/MaxPool_1MaxPool5sequential/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????`*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d/MaxPool_1?
*sequential/conv2_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype02,
*sequential/conv2_1/Conv2D_1/ReadVariableOp?
sequential/conv2_1/Conv2D_1Conv2D+sequential/max_pooling2d/MaxPool_1:output:02sequential/conv2_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv2_1/Conv2D_1?
+sequential/conv2_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv2_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv2_1/BiasAdd_1/ReadVariableOp?
sequential/conv2_1/BiasAdd_1BiasAdd$sequential/conv2_1/Conv2D_1:output:03sequential/conv2_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/BiasAdd_1?
sequential/conv2_1/Relu_1Relu%sequential/conv2_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv2_1/Relu_1?
1sequential/batch_normalization_1/ReadVariableOp_2ReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_2?
1sequential/batch_normalization_1/ReadVariableOp_3ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_3?
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource0^sequential/batch_normalization_1/AssignNewValue*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource2^sequential/batch_normalization_1/AssignNewValue_1*
_output_shapes
:*
dtype02F
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?
3sequential/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3'sequential/conv2_1/Relu_1:activations:09sequential/batch_normalization_1/ReadVariableOp_2:value:09sequential/batch_normalization_1/ReadVariableOp_3:value:0Jsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Lsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=25
3sequential/batch_normalization_1/FusedBatchNormV3_1?
1sequential/batch_normalization_1/AssignNewValue_2AssignVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource@sequential/batch_normalization_1/FusedBatchNormV3_1:batch_mean:00^sequential/batch_normalization_1/AssignNewValueC^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential/batch_normalization_1/AssignNewValue_2?
1sequential/batch_normalization_1/AssignNewValue_3AssignVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceDsequential/batch_normalization_1/FusedBatchNormV3_1:batch_variance:02^sequential/batch_normalization_1/AssignNewValue_1E^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype023
1sequential/batch_normalization_1/AssignNewValue_3?
$sequential/max_pooling2d_1/MaxPool_1MaxPool7sequential/batch_normalization_1/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2&
$sequential/max_pooling2d_1/MaxPool_1?
*sequential/conv3_1/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_1/Conv2D_1/ReadVariableOp?
sequential/conv3_1/Conv2D_1Conv2D-sequential/max_pooling2d_1/MaxPool_1:output:02sequential/conv3_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
sequential/conv3_1/Conv2D_1?
+sequential/conv3_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_1/BiasAdd_1/ReadVariableOp?
sequential/conv3_1/BiasAdd_1BiasAdd$sequential/conv3_1/Conv2D_1:output:03sequential/conv3_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/BiasAdd_1?
sequential/conv3_1/Relu_1Relu%sequential/conv3_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????		?2
sequential/conv3_1/Relu_1?
*sequential/conv3_2/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_2/Conv2D_1/ReadVariableOp?
sequential/conv3_2/Conv2D_1Conv2D'sequential/conv3_1/Relu_1:activations:02sequential/conv3_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_2/Conv2D_1?
+sequential/conv3_2/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_2/BiasAdd_1/ReadVariableOp?
sequential/conv3_2/BiasAdd_1BiasAdd$sequential/conv3_2/Conv2D_1:output:03sequential/conv3_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/BiasAdd_1?
sequential/conv3_2/Relu_1Relu%sequential/conv3_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_2/Relu_1?
*sequential/conv3_3/Conv2D_1/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*sequential/conv3_3/Conv2D_1/ReadVariableOp?
sequential/conv3_3/Conv2D_1Conv2D'sequential/conv3_2/Relu_1:activations:02sequential/conv3_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential/conv3_3/Conv2D_1?
+sequential/conv3_3/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_conv3_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/conv3_3/BiasAdd_1/ReadVariableOp?
sequential/conv3_3/BiasAdd_1BiasAdd$sequential/conv3_3/Conv2D_1:output:03sequential/conv3_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/BiasAdd_1?
sequential/conv3_3/Relu_1Relu%sequential/conv3_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
sequential/conv3_3/Relu_1?
$sequential/max_pooling2d_2/MaxPool_1MaxPool'sequential/conv3_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2&
$sequential/max_pooling2d_2/MaxPool_1?
"sequential/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2$
"sequential/dropout/dropout_1/Const?
 sequential/dropout/dropout_1/MulMul-sequential/max_pooling2d_2/MaxPool_1:output:0+sequential/dropout/dropout_1/Const:output:0*
T0*0
_output_shapes
:??????????2"
 sequential/dropout/dropout_1/Mul?
"sequential/dropout/dropout_1/ShapeShape-sequential/max_pooling2d_2/MaxPool_1:output:0*
T0*
_output_shapes
:2$
"sequential/dropout/dropout_1/Shape?
9sequential/dropout/dropout_1/random_uniform/RandomUniformRandomUniform+sequential/dropout/dropout_1/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02;
9sequential/dropout/dropout_1/random_uniform/RandomUniform?
+sequential/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+sequential/dropout/dropout_1/GreaterEqual/y?
)sequential/dropout/dropout_1/GreaterEqualGreaterEqualBsequential/dropout/dropout_1/random_uniform/RandomUniform:output:04sequential/dropout/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2+
)sequential/dropout/dropout_1/GreaterEqual?
!sequential/dropout/dropout_1/CastCast-sequential/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2#
!sequential/dropout/dropout_1/Cast?
"sequential/dropout/dropout_1/Mul_1Mul$sequential/dropout/dropout_1/Mul:z:0%sequential/dropout/dropout_1/Cast:y:0*
T0*0
_output_shapes
:??????????2$
"sequential/dropout/dropout_1/Mul_1?
sequential/Flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/Flatten/Const_1?
sequential/Flatten/Reshape_1Reshape&sequential/dropout/dropout_1/Mul_1:z:0#sequential/Flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/Flatten/Reshape_1?
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp?
sequential/dense/MatMul_1MatMul%sequential/Flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul_1?
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp?
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd_1?
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu_1?
$sequential/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$sequential/dropout_1/dropout_1/Const?
"sequential/dropout_1/dropout_1/MulMul%sequential/dense/Relu_1:activations:0-sequential/dropout_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2$
"sequential/dropout_1/dropout_1/Mul?
$sequential/dropout_1/dropout_1/ShapeShape%sequential/dense/Relu_1:activations:0*
T0*
_output_shapes
:2&
$sequential/dropout_1/dropout_1/Shape?
;sequential/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform-sequential/dropout_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02=
;sequential/dropout_1/dropout_1/random_uniform/RandomUniform?
-sequential/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential/dropout_1/dropout_1/GreaterEqual/y?
+sequential/dropout_1/dropout_1/GreaterEqualGreaterEqualDsequential/dropout_1/dropout_1/random_uniform/RandomUniform:output:06sequential/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/dropout_1/dropout_1/GreaterEqual?
#sequential/dropout_1/dropout_1/CastCast/sequential/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2%
#sequential/dropout_1/dropout_1/Cast?
$sequential/dropout_1/dropout_1/Mul_1Mul&sequential/dropout_1/dropout_1/Mul:z:0'sequential/dropout_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2&
$sequential/dropout_1/dropout_1/Mul_1?
*sequential/dense_1/MatMul_1/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential/dense_1/MatMul_1/ReadVariableOp?
sequential/dense_1/MatMul_1MatMul(sequential/dropout_1/dropout_1/Mul_1:z:02sequential/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul_1?
+sequential/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential/dense_1/BiasAdd_1/ReadVariableOp?
sequential/dense_1/BiasAdd_1BiasAdd%sequential/dense_1/MatMul_1:product:03sequential/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd_1?
sequential/dense_1/Relu_1Relu%sequential/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu_1?

lambda/subSub%sequential/dense_1/Relu:activations:0'sequential/dense_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2

lambda/subk
lambda/SquareSquarelambda/sub:z:0*
T0*(
_output_shapes
:??????????2
lambda/Square~
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Sum/reduction_indices?

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

lambda/Suma
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/Const?
lambda/MaximumMaximumlambda/Sum:output:0lambda/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda/Maximumh
lambda/SqrtSqrtlambda/Maximum:z:0*
T0*'
_output_shapes
:?????????2
lambda/Sqrt?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv2_1_conv2d_readvariableop_resource*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_conv3_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/Sqrt:y:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_10^sequential/batch_normalization/AssignNewValue_20^sequential/batch_normalization/AssignNewValue_3?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1A^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOpC^sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_10^sequential/batch_normalization/ReadVariableOp_20^sequential/batch_normalization/ReadVariableOp_30^sequential/batch_normalization_1/AssignNewValue2^sequential/batch_normalization_1/AssignNewValue_12^sequential/batch_normalization_1/AssignNewValue_22^sequential/batch_normalization_1/AssignNewValue_3A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1C^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpE^sequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_12^sequential/batch_normalization_1/ReadVariableOp_22^sequential/batch_normalization_1/ReadVariableOp_3*^sequential/conv1_1/BiasAdd/ReadVariableOp,^sequential/conv1_1/BiasAdd_1/ReadVariableOp)^sequential/conv1_1/Conv2D/ReadVariableOp+^sequential/conv1_1/Conv2D_1/ReadVariableOp*^sequential/conv2_1/BiasAdd/ReadVariableOp,^sequential/conv2_1/BiasAdd_1/ReadVariableOp)^sequential/conv2_1/Conv2D/ReadVariableOp+^sequential/conv2_1/Conv2D_1/ReadVariableOp*^sequential/conv3_1/BiasAdd/ReadVariableOp,^sequential/conv3_1/BiasAdd_1/ReadVariableOp)^sequential/conv3_1/Conv2D/ReadVariableOp+^sequential/conv3_1/Conv2D_1/ReadVariableOp*^sequential/conv3_2/BiasAdd/ReadVariableOp,^sequential/conv3_2/BiasAdd_1/ReadVariableOp)^sequential/conv3_2/Conv2D/ReadVariableOp+^sequential/conv3_2/Conv2D_1/ReadVariableOp*^sequential/conv3_3/BiasAdd/ReadVariableOp,^sequential/conv3_3/BiasAdd_1/ReadVariableOp)^sequential/conv3_3/Conv2D/ReadVariableOp+^sequential/conv3_3/Conv2D_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/BiasAdd_1/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp+^sequential/dense_1/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12b
/sequential/batch_normalization/AssignNewValue_2/sequential/batch_normalization/AssignNewValue_22b
/sequential/batch_normalization/AssignNewValue_3/sequential/batch_normalization/AssignNewValue_32?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp@sequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2?
Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Bsequential/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12b
/sequential/batch_normalization/ReadVariableOp_2/sequential/batch_normalization/ReadVariableOp_22b
/sequential/batch_normalization/ReadVariableOp_3/sequential/batch_normalization/ReadVariableOp_32b
/sequential/batch_normalization_1/AssignNewValue/sequential/batch_normalization_1/AssignNewValue2f
1sequential/batch_normalization_1/AssignNewValue_11sequential/batch_normalization_1/AssignNewValue_12f
1sequential/batch_normalization_1/AssignNewValue_21sequential/batch_normalization_1/AssignNewValue_22f
1sequential/batch_normalization_1/AssignNewValue_31sequential/batch_normalization_1/AssignNewValue_32?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Bsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpBsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Dsequential/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12f
1sequential/batch_normalization_1/ReadVariableOp_21sequential/batch_normalization_1/ReadVariableOp_22f
1sequential/batch_normalization_1/ReadVariableOp_31sequential/batch_normalization_1/ReadVariableOp_32V
)sequential/conv1_1/BiasAdd/ReadVariableOp)sequential/conv1_1/BiasAdd/ReadVariableOp2Z
+sequential/conv1_1/BiasAdd_1/ReadVariableOp+sequential/conv1_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv1_1/Conv2D/ReadVariableOp(sequential/conv1_1/Conv2D/ReadVariableOp2X
*sequential/conv1_1/Conv2D_1/ReadVariableOp*sequential/conv1_1/Conv2D_1/ReadVariableOp2V
)sequential/conv2_1/BiasAdd/ReadVariableOp)sequential/conv2_1/BiasAdd/ReadVariableOp2Z
+sequential/conv2_1/BiasAdd_1/ReadVariableOp+sequential/conv2_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv2_1/Conv2D/ReadVariableOp(sequential/conv2_1/Conv2D/ReadVariableOp2X
*sequential/conv2_1/Conv2D_1/ReadVariableOp*sequential/conv2_1/Conv2D_1/ReadVariableOp2V
)sequential/conv3_1/BiasAdd/ReadVariableOp)sequential/conv3_1/BiasAdd/ReadVariableOp2Z
+sequential/conv3_1/BiasAdd_1/ReadVariableOp+sequential/conv3_1/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_1/Conv2D/ReadVariableOp(sequential/conv3_1/Conv2D/ReadVariableOp2X
*sequential/conv3_1/Conv2D_1/ReadVariableOp*sequential/conv3_1/Conv2D_1/ReadVariableOp2V
)sequential/conv3_2/BiasAdd/ReadVariableOp)sequential/conv3_2/BiasAdd/ReadVariableOp2Z
+sequential/conv3_2/BiasAdd_1/ReadVariableOp+sequential/conv3_2/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_2/Conv2D/ReadVariableOp(sequential/conv3_2/Conv2D/ReadVariableOp2X
*sequential/conv3_2/Conv2D_1/ReadVariableOp*sequential/conv3_2/Conv2D_1/ReadVariableOp2V
)sequential/conv3_3/BiasAdd/ReadVariableOp)sequential/conv3_3/BiasAdd/ReadVariableOp2Z
+sequential/conv3_3/BiasAdd_1/ReadVariableOp+sequential/conv3_3/BiasAdd_1/ReadVariableOp2T
(sequential/conv3_3/Conv2D/ReadVariableOp(sequential/conv3_3/Conv2D/ReadVariableOp2X
*sequential/conv3_3/Conv2D_1/ReadVariableOp*sequential/conv3_3/Conv2D_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/BiasAdd_1/ReadVariableOp+sequential/dense_1/BiasAdd_1/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2X
*sequential/dense_1/MatMul_1/ReadVariableOp*sequential/dense_1/MatMul_1/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35512516

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88`:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88`
 
_user_specified_nameinputs
?
?
*__inference_conv3_3_layer_call_fn_35515668

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_3_layer_call_and_return_conditional_losses_355126412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515310

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+?????????8??????????????????:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_35512451

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_layer_call_fn_35515695

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_355128752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv1_1_layer_call_and_return_conditional_losses_35515283

inputs8
conv2d_readvariableop_resource:		`-
biasadd_readvariableop_resource:`
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88`2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????88`2
Relu?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv1_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????88`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35512203

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+?????????8??????????????????:8:8:8:8:*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+?????????8??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+?????????8??????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+?????????8??????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_35515852U
9conv3_3_kernel_regularizer_square_readvariableop_resource:??
identity??0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv3_3_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
IdentityIdentity"conv3_3/kernel/Regularizer/mul:z:01^conv3_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515466

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
data_formatNCHW*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?c
?

C__inference_model_layer_call_and_return_conditional_losses_35513612

inputs
inputs_1-
sequential_35513487:		`!
sequential_35513489:`!
sequential_35513491:8!
sequential_35513493:8!
sequential_35513495:8!
sequential_35513497:8.
sequential_35513499:`?"
sequential_35513501:	?!
sequential_35513503:!
sequential_35513505:!
sequential_35513507:!
sequential_35513509:/
sequential_35513511:??"
sequential_35513513:	?/
sequential_35513515:??"
sequential_35513517:	?/
sequential_35513519:??"
sequential_35513521:	?'
sequential_35513523:
??"
sequential_35513525:	?'
sequential_35513527:
??"
sequential_35513529:	?
identity??0conv1_1/kernel/Regularizer/Square/ReadVariableOp?0conv2_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?0conv3_2/kernel/Regularizer/Square/ReadVariableOp?0conv3_3/kernel/Regularizer/Square/ReadVariableOp?.dense/kernel/Regularizer/Square/ReadVariableOp?0dense_1/kernel/Regularizer/Square/ReadVariableOp?"sequential/StatefulPartitionedCall?$sequential/StatefulPartitionedCall_1?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_35513487sequential_35513489sequential_35513491sequential_35513493sequential_35513495sequential_35513497sequential_35513499sequential_35513501sequential_35513503sequential_35513505sequential_35513507sequential_35513509sequential_35513511sequential_35513513sequential_35513515sequential_35513517sequential_35513519sequential_35513521sequential_35513523sequential_35513525sequential_35513527sequential_35513529*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592$
"sequential/StatefulPartitionedCall?
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_35513487sequential_35513489sequential_35513491sequential_35513493sequential_35513495sequential_35513497sequential_35513499sequential_35513501sequential_35513503sequential_35513505sequential_35513507sequential_35513509sequential_35513511sequential_35513513sequential_35513515sequential_35513517sequential_35513519sequential_35513521sequential_35513523sequential_35513525sequential_35513527sequential_35513529*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_355127592&
$sequential/StatefulPartitionedCall_1?
lambda/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_355135672
lambda/PartitionedCall?
0conv1_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513487*&
_output_shapes
:		`*
dtype022
0conv1_1/kernel/Regularizer/Square/ReadVariableOp?
!conv1_1/kernel/Regularizer/SquareSquare8conv1_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		`2#
!conv1_1/kernel/Regularizer/Square?
 conv1_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv1_1/kernel/Regularizer/Const?
conv1_1/kernel/Regularizer/SumSum%conv1_1/kernel/Regularizer/Square:y:0)conv1_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/Sum?
 conv1_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv1_1/kernel/Regularizer/mul/x?
conv1_1/kernel/Regularizer/mulMul)conv1_1/kernel/Regularizer/mul/x:output:0'conv1_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv1_1/kernel/Regularizer/mul?
0conv2_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513499*'
_output_shapes
:`?*
dtype022
0conv2_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2_1/kernel/Regularizer/SquareSquare8conv2_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:`?2#
!conv2_1/kernel/Regularizer/Square?
 conv2_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv2_1/kernel/Regularizer/Const?
conv2_1/kernel/Regularizer/SumSum%conv2_1/kernel/Regularizer/Square:y:0)conv2_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/Sum?
 conv2_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv2_1/kernel/Regularizer/mul/x?
conv2_1/kernel/Regularizer/mulMul)conv2_1/kernel/Regularizer/mul/x:output:0'conv2_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2_1/kernel/Regularizer/mul?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513511*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
0conv3_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513515*(
_output_shapes
:??*
dtype022
0conv3_2/kernel/Regularizer/Square/ReadVariableOp?
!conv3_2/kernel/Regularizer/SquareSquare8conv3_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_2/kernel/Regularizer/Square?
 conv3_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_2/kernel/Regularizer/Const?
conv3_2/kernel/Regularizer/SumSum%conv3_2/kernel/Regularizer/Square:y:0)conv3_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/Sum?
 conv3_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_2/kernel/Regularizer/mul/x?
conv3_2/kernel/Regularizer/mulMul)conv3_2/kernel/Regularizer/mul/x:output:0'conv3_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_2/kernel/Regularizer/mul?
0conv3_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513519*(
_output_shapes
:??*
dtype022
0conv3_3/kernel/Regularizer/Square/ReadVariableOp?
!conv3_3/kernel/Regularizer/SquareSquare8conv3_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_3/kernel/Regularizer/Square?
 conv3_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_3/kernel/Regularizer/Const?
conv3_3/kernel/Regularizer/SumSum%conv3_3/kernel/Regularizer/Square:y:0)conv3_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/Sum?
 conv3_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_3/kernel/Regularizer/mul/x?
conv3_3/kernel/Regularizer/mulMul)conv3_3/kernel/Regularizer/mul/x:output:0'conv3_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_3/kernel/Regularizer/mul?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513523* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_35513527* 
_output_shapes
:
??*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp?
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_1/kernel/Regularizer/Square?
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const?
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum?
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 dense_1/kernel/Regularizer/mul/x?
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul?
IdentityIdentitylambda/PartitionedCall:output:01^conv1_1/kernel/Regularizer/Square/ReadVariableOp1^conv2_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp1^conv3_2/kernel/Regularizer/Square/ReadVariableOp1^conv3_3/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:?????????@@:?????????@@: : : : : : : : : : : : : : : : : : : : : : 2d
0conv1_1/kernel/Regularizer/Square/ReadVariableOp0conv1_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv2_1/kernel/Regularizer/Square/ReadVariableOp0conv2_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_2/kernel/Regularizer/Square/ReadVariableOp0conv3_2/kernel/Regularizer/Square/ReadVariableOp2d
0conv3_3/kernel/Regularizer/Square/ReadVariableOp0conv3_3/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_35512463

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_35515830U
9conv3_1_kernel_regularizer_square_readvariableop_resource:??
identity??0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv3_1_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
IdentityIdentity"conv3_1/kernel/Regularizer/mul:z:01^conv3_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_5_35515863K
7dense_kernel_regularizer_square_readvariableop_resource:
??
identity??.dense/kernel/Regularizer/Square/ReadVariableOp?
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp?
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2!
dense/kernel/Regularizer/Square?
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const?
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum?
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2 
dense/kernel/Regularizer/mul/x?
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul?
IdentityIdentity dense/kernel/Regularizer/mul:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
?
a
E__inference_Flatten_layer_call_and_return_conditional_losses_35512661

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35512941

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515520

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
data_formatNCHW*
epsilon%??'7*
exponential_avg_factor%???=2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv3_1_layer_call_fn_35515604

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv3_1_layer_call_and_return_conditional_losses_355125952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_35515559

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_355125672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv3_1_layer_call_and_return_conditional_losses_35512595

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
0conv3_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv3_1/kernel/Regularizer/Square/ReadVariableOp?
!conv3_1/kernel/Regularizer/SquareSquare8conv3_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:??2#
!conv3_1/kernel/Regularizer/Square?
 conv3_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 conv3_1/kernel/Regularizer/Const?
conv3_1/kernel/Regularizer/SumSum%conv3_1/kernel/Regularizer/Square:y:0)conv3_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/Sum?
 conv3_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92"
 conv3_1/kernel/Regularizer/mul/x?
conv3_1/kernel/Regularizer/mulMul)conv3_1/kernel/Regularizer/mul/x:output:0'conv3_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv3_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp1^conv3_1/kernel/Regularizer/Square/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2d
0conv3_1/kernel/Regularizer/Square/ReadVariableOp0conv3_1/kernel/Regularizer/Square/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@
C
input_28
serving_default_input_2:0?????????@@:
lambda0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "Flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2304, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQNaBXZlY3Rz2gF42gF5qQByCgAAAPofPGlweXRob24taW5wdXQtMjQtZjFiZWQ0ZDQ5Y2Qw\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMjQtZjFiZWQ0ZDQ5Y2QwPtoTZXVjbF9kaXN0YW5jZV9zaGFwZQUAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "shared_object_id": 42, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, {"class_name": "TensorShape", "items": [null, 64, 64, 3]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "float32", "input_2"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "Flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2304, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQNaBXZlY3Rz2gF42gF5qQByCgAAAPofPGlweXRob24taW5wdXQtMjQtZjFiZWQ0ZDQ5Y2Qw\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMjQtZjFiZWQ0ZDQ5Y2QwPtoTZXVjbF9kaXN0YW5jZV9zaGFwZQUAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "name": "lambda", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]], "shared_object_id": 41}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["lambda", 0, 0]]}}, "training_config": {"loss": "contrastive_loss", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 45}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-08, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
??
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
layer_with_weights-6
layer-8
layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_sequential̀{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "Flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2304, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "float32", "conv1_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1_1_input"}, "shared_object_id": 2}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 10}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11}, {"class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 13}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 19}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 20}, {"class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 22}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 25}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26}, {"class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 28}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 30}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 31}, {"class_name": "Flatten", "config": {"name": "Flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2304, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}, "shared_object_id": 38}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39}]}}}
?

	variables
regularization_losses
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAcAAABDAAAAcygAAAB8AFwCfQF9AnQAoAF0AGoCdACgA3wBfAIYAKEBZAFk\nAmQDjQOhAVMAKQRO6QEAAABUKQLaBGF4aXPaCGtlZXBkaW1zKQTaAUvaBHNxcnTaA3N1bdoGc3F1\nYXJlKQNaBXZlY3Rz2gF42gF5qQByCgAAAPofPGlweXRob24taW5wdXQtMjQtZjFiZWQ0ZDQ5Y2Qw\nPtoSZXVjbGlkZWFuX2Rpc3RhbmNlAQAAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAwAAAAIAAABDAAAAcxQAAAB8AFwCfQF9AnwBZAEZAGQCZgJTACkDTukAAAAA6QEA\nAACpACkD2gZzaGFwZXPaBnNoYXBlMdoGc2hhcGUycgMAAAByAwAAAPofPGlweXRob24taW5wdXQt\nMjQtZjFiZWQ0ZDQ5Y2QwPtoTZXVjbF9kaXN0YW5jZV9zaGFwZQUAAABzBAAAAAABCAE=\n", null, null]}, "output_shape_type": "lambda", "output_shape_module": "__main__", "arguments": {}}, "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]], "shared_object_id": 41}
?
"iter
	#decay
$learning_rate
%momentum
&rho
'rms?
(rms?
)rms?
*rms?
-rms?
.rms?
/rms?
0rms?
3rms?
4rms?
5rms?
6rms?
7rms?
8rms?
9rms?
:rms?
;rms?
<rms?"
	optimizer
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'0
(1
)2
*3
-4
.5
/6
07
38
49
510
611
712
813
914
:15
;16
<17"
trackable_list_wrapper
?
	variables
=metrics
regularization_losses
>layer_regularization_losses

?layers
@non_trainable_variables
trainable_variables
Alayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

'kernel
(bias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?

Faxis
	)gamma
*beta
+moving_mean
,moving_variance
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 56}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 96]}}
?
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 48}}
?

-kernel
.bias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 13}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 27, 96]}}
?

Saxis
	/gamma
0beta
1moving_mean
2moving_variance
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 23}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 256]}}
?
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}}
?

3kernel
4bias
\	variables
]regularization_losses
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 22}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 256]}}
?

5kernel
6bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 25}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 384]}}
?

7kernel
8bias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv3_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}, "shared_object_id": 28}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 384]}}
?
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 55}}
?
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 31}
?
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "Flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 56}}
?

9kernel
:bias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2304, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}, "shared_object_id": 34}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}, "shared_object_id": 57}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 36}
?

;kernel
<bias
|	variables
}regularization_losses
~trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": "initialize_weights", "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0005000000237487257}, "shared_object_id": 38}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21"
trackable_list_wrapper
X
?0
?1
?2
?3
?4
?5
?6"
trackable_list_wrapper
?
'0
(1
)2
*3
-4
.5
/6
07
38
49
510
611
712
813
914
:15
;16
<17"
trackable_list_wrapper
?
	variables
?metrics
regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
trainable_variables
?layer_metrics
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
	variables
?metrics
regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
 trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
(:&		`2conv1_1/kernel
:`2conv1_1/bias
':%82batch_normalization/gamma
&:$82batch_normalization/beta
/:-8 (2batch_normalization/moving_mean
3:18 (2#batch_normalization/moving_variance
):'`?2conv2_1/kernel
:?2conv2_1/bias
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
*:(??2conv3_1/kernel
:?2conv3_1/bias
*:(??2conv3_2/kernel
:?2conv3_2/bias
*:(??2conv3_3/kernel
:?2conv3_3/bias
 :
??2dense/kernel
:?2
dense/bias
": 
??2dense_1/kernel
:?2dense_1/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
+0
,1
12
23"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
'0
(1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
B	variables
?metrics
Cregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Dtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
G	variables
?metrics
Hregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Itrainable_variables
?layer_metrics
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
K	variables
?metrics
Lregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Mtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
O	variables
?metrics
Pregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Qtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
T	variables
?metrics
Uregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Vtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
X	variables
?metrics
Yregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
Ztrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
\	variables
?metrics
]regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
^trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
`	variables
?metrics
aregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
btrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
d	variables
?metrics
eregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ftrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
h	variables
?metrics
iregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
jtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
l	variables
?metrics
mregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ntrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
p	variables
?metrics
qregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
rtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
t	variables
?metrics
uregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
vtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
x	variables
?metrics
yregularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
ztrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
|	variables
?metrics
}regularization_losses
 ?layer_regularization_losses
?layers
?non_trainable_variables
~trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14"
trackable_list_wrapper
<
+0
,1
12
23"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 59}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 45}
 "
trackable_list_wrapper
(
?0"
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
.
+0
,1"
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
(
?0"
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
.
10
21"
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
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
(
?0"
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0		`2RMSprop/conv1_1/kernel/rms
$:"`2RMSprop/conv1_1/bias/rms
1:/82%RMSprop/batch_normalization/gamma/rms
0:.82$RMSprop/batch_normalization/beta/rms
3:1`?2RMSprop/conv2_1/kernel/rms
%:#?2RMSprop/conv2_1/bias/rms
3:12'RMSprop/batch_normalization_1/gamma/rms
2:02&RMSprop/batch_normalization_1/beta/rms
4:2??2RMSprop/conv3_1/kernel/rms
%:#?2RMSprop/conv3_1/bias/rms
4:2??2RMSprop/conv3_2/kernel/rms
%:#?2RMSprop/conv3_2/bias/rms
4:2??2RMSprop/conv3_3/kernel/rms
%:#?2RMSprop/conv3_3/bias/rms
*:(
??2RMSprop/dense/kernel/rms
#:!?2RMSprop/dense/bias/rms
,:*
??2RMSprop/dense_1/kernel/rms
%:#?2RMSprop/dense_1/bias/rms
?2?
#__inference__wrapped_model_35512181?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *^?[
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
?2?
C__inference_model_layer_call_and_return_conditional_losses_35514482
C__inference_model_layer_call_and_return_conditional_losses_35514710
C__inference_model_layer_call_and_return_conditional_losses_35514066
C__inference_model_layer_call_and_return_conditional_losses_35514182?
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
?2?
(__inference_model_layer_call_fn_35513659
(__inference_model_layer_call_fn_35514760
(__inference_model_layer_call_fn_35514810
(__inference_model_layer_call_fn_35513950?
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
H__inference_sequential_layer_call_and_return_conditional_losses_35514982
H__inference_sequential_layer_call_and_return_conditional_losses_35515126
H__inference_sequential_layer_call_and_return_conditional_losses_35513373
H__inference_sequential_layer_call_and_return_conditional_losses_35513478?
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
?2?
-__inference_sequential_layer_call_fn_35512806
-__inference_sequential_layer_call_fn_35515175
-__inference_sequential_layer_call_fn_35515224
-__inference_sequential_layer_call_fn_35513268?
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
?2?
D__inference_lambda_layer_call_and_return_conditional_losses_35515236
D__inference_lambda_layer_call_and_return_conditional_losses_35515248?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_lambda_layer_call_fn_35515254
)__inference_lambda_layer_call_fn_35515260?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_35514282input_1input_2"?
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
?2?
E__inference_conv1_1_layer_call_and_return_conditional_losses_35515283?
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
*__inference_conv1_1_layer_call_fn_35515292?
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
?2?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515310
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515328
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515346
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515364?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_layer_call_fn_35515377
6__inference_batch_normalization_layer_call_fn_35515390
6__inference_batch_normalization_layer_call_fn_35515403
6__inference_batch_normalization_layer_call_fn_35515416?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_35512313?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_layer_call_fn_35512319?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2_1_layer_call_and_return_conditional_losses_35515439?
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
*__inference_conv2_1_layer_call_fn_35515448?
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
?2?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515466
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515484
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515502
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515520?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_1_layer_call_fn_35515533
8__inference_batch_normalization_1_layer_call_fn_35515546
8__inference_batch_normalization_1_layer_call_fn_35515559
8__inference_batch_normalization_1_layer_call_fn_35515572?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_35512451?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_1_layer_call_fn_35512457?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv3_1_layer_call_and_return_conditional_losses_35515595?
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
*__inference_conv3_1_layer_call_fn_35515604?
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
E__inference_conv3_2_layer_call_and_return_conditional_losses_35515627?
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
*__inference_conv3_2_layer_call_fn_35515636?
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
E__inference_conv3_3_layer_call_and_return_conditional_losses_35515659?
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
*__inference_conv3_3_layer_call_fn_35515668?
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
?2?
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_35512463?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_max_pooling2d_2_layer_call_fn_35512469?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_layer_call_and_return_conditional_losses_35515673
E__inference_dropout_layer_call_and_return_conditional_losses_35515685?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_layer_call_fn_35515690
*__inference_dropout_layer_call_fn_35515695?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_Flatten_layer_call_and_return_conditional_losses_35515701?
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
*__inference_Flatten_layer_call_fn_35515706?
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
C__inference_dense_layer_call_and_return_conditional_losses_35515729?
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
(__inference_dense_layer_call_fn_35515738?
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
?2?
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515743
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515755?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_1_layer_call_fn_35515760
,__inference_dropout_1_layer_call_fn_35515765?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_35515788?
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
*__inference_dense_1_layer_call_fn_35515797?
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
__inference_loss_fn_0_35515808?
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
annotations? *? 
?2?
__inference_loss_fn_1_35515819?
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
annotations? *? 
?2?
__inference_loss_fn_2_35515830?
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
annotations? *? 
?2?
__inference_loss_fn_3_35515841?
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
annotations? *? 
?2?
__inference_loss_fn_4_35515852?
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
annotations? *? 
?2?
__inference_loss_fn_5_35515863?
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
annotations? *? 
?2?
__inference_loss_fn_6_35515874?
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
annotations? *? ?
E__inference_Flatten_layer_call_and_return_conditional_losses_35515701b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
*__inference_Flatten_layer_call_fn_35515706U8?5
.?+
)?&
inputs??????????
? "????????????
#__inference__wrapped_model_35512181?'()*+,-./0123456789:;<h?e
^?[
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
? "/?,
*
lambda ?
lambda??????????
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515466?/012M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515484?/012M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515502t/012<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35515520t/012<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
8__inference_batch_normalization_1_layer_call_fn_35515533?/012M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_1_layer_call_fn_35515546?/012M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_1_layer_call_fn_35515559g/012<?9
2?/
)?&
inputs??????????
p 
? "!????????????
8__inference_batch_normalization_1_layer_call_fn_35515572g/012<?9
2?/
)?&
inputs??????????
p
? "!????????????
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515310?)*+,M?J
C?@
:?7
inputs+?????????8??????????????????
p 
? "??<
5?2
0+?????????8??????????????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515328?)*+,M?J
C?@
:?7
inputs+?????????8??????????????????
p
? "??<
5?2
0+?????????8??????????????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515346r)*+,;?8
1?.
(?%
inputs?????????88`
p 
? "-?*
#? 
0?????????88`
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_35515364r)*+,;?8
1?.
(?%
inputs?????????88`
p
? "-?*
#? 
0?????????88`
? ?
6__inference_batch_normalization_layer_call_fn_35515377?)*+,M?J
C?@
:?7
inputs+?????????8??????????????????
p 
? "2?/+?????????8???????????????????
6__inference_batch_normalization_layer_call_fn_35515390?)*+,M?J
C?@
:?7
inputs+?????????8??????????????????
p
? "2?/+?????????8???????????????????
6__inference_batch_normalization_layer_call_fn_35515403e)*+,;?8
1?.
(?%
inputs?????????88`
p 
? " ??????????88`?
6__inference_batch_normalization_layer_call_fn_35515416e)*+,;?8
1?.
(?%
inputs?????????88`
p
? " ??????????88`?
E__inference_conv1_1_layer_call_and_return_conditional_losses_35515283l'(7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????88`
? ?
*__inference_conv1_1_layer_call_fn_35515292_'(7?4
-?*
(?%
inputs?????????@@
? " ??????????88`?
E__inference_conv2_1_layer_call_and_return_conditional_losses_35515439m-.7?4
-?*
(?%
inputs?????????`
? ".?+
$?!
0??????????
? ?
*__inference_conv2_1_layer_call_fn_35515448`-.7?4
-?*
(?%
inputs?????????`
? "!????????????
E__inference_conv3_1_layer_call_and_return_conditional_losses_35515595n348?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????		?
? ?
*__inference_conv3_1_layer_call_fn_35515604a348?5
.?+
)?&
inputs??????????
? "!??????????		??
E__inference_conv3_2_layer_call_and_return_conditional_losses_35515627n568?5
.?+
)?&
inputs?????????		?
? ".?+
$?!
0??????????
? ?
*__inference_conv3_2_layer_call_fn_35515636a568?5
.?+
)?&
inputs?????????		?
? "!????????????
E__inference_conv3_3_layer_call_and_return_conditional_losses_35515659n788?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv3_3_layer_call_fn_35515668a788?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_dense_1_layer_call_and_return_conditional_losses_35515788^;<0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_1_layer_call_fn_35515797Q;<0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_layer_call_and_return_conditional_losses_35515729^9:0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_layer_call_fn_35515738Q9:0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515743^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
G__inference_dropout_1_layer_call_and_return_conditional_losses_35515755^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
,__inference_dropout_1_layer_call_fn_35515760Q4?1
*?'
!?
inputs??????????
p 
? "????????????
,__inference_dropout_1_layer_call_fn_35515765Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_layer_call_and_return_conditional_losses_35515673n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
E__inference_dropout_layer_call_and_return_conditional_losses_35515685n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
*__inference_dropout_layer_call_fn_35515690a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
*__inference_dropout_layer_call_fn_35515695a<?9
2?/
)?&
inputs??????????
p
? "!????????????
D__inference_lambda_layer_call_and_return_conditional_losses_35515236?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p 
? "%?"
?
0?????????
? ?
D__inference_lambda_layer_call_and_return_conditional_losses_35515248?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p
? "%?"
?
0?????????
? ?
)__inference_lambda_layer_call_fn_35515254?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p 
? "???????????
)__inference_lambda_layer_call_fn_35515260?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p
? "??????????=
__inference_loss_fn_0_35515808'?

? 
? "? =
__inference_loss_fn_1_35515819-?

? 
? "? =
__inference_loss_fn_2_355158303?

? 
? "? =
__inference_loss_fn_3_355158415?

? 
? "? =
__inference_loss_fn_4_355158527?

? 
? "? =
__inference_loss_fn_5_355158639?

? 
? "? =
__inference_loss_fn_6_35515874;?

? 
? "? ?
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_35512451?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_1_layer_call_fn_35512457?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_35512463?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_2_layer_call_fn_35512469?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_35512313?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_layer_call_fn_35512319?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_layer_call_and_return_conditional_losses_35514066?'()*+,-./0123456789:;<p?m
f?c
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_35514182?'()*+,-./0123456789:;<p?m
f?c
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
p

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_35514482?'()*+,-./0123456789:;<r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_35514710?'()*+,-./0123456789:;<r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p

 
? "%?"
?
0?????????
? ?
(__inference_model_layer_call_fn_35513659?'()*+,-./0123456789:;<p?m
f?c
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
p 

 
? "???????????
(__inference_model_layer_call_fn_35513950?'()*+,-./0123456789:;<p?m
f?c
Y?V
)?&
input_1?????????@@
)?&
input_2?????????@@
p

 
? "???????????
(__inference_model_layer_call_fn_35514760?'()*+,-./0123456789:;<r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p 

 
? "???????????
(__inference_model_layer_call_fn_35514810?'()*+,-./0123456789:;<r?o
h?e
[?X
*?'
inputs/0?????????@@
*?'
inputs/1?????????@@
p

 
? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_35513373?'()*+,-./0123456789:;<F?C
<?9
/?,
conv1_1_input?????????@@
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_35513478?'()*+,-./0123456789:;<F?C
<?9
/?,
conv1_1_input?????????@@
p

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_35514982?'()*+,-./0123456789:;<??<
5?2
(?%
inputs?????????@@
p 

 
? "&?#
?
0??????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_35515126?'()*+,-./0123456789:;<??<
5?2
(?%
inputs?????????@@
p

 
? "&?#
?
0??????????
? ?
-__inference_sequential_layer_call_fn_35512806{'()*+,-./0123456789:;<F?C
<?9
/?,
conv1_1_input?????????@@
p 

 
? "????????????
-__inference_sequential_layer_call_fn_35513268{'()*+,-./0123456789:;<F?C
<?9
/?,
conv1_1_input?????????@@
p

 
? "????????????
-__inference_sequential_layer_call_fn_35515175t'()*+,-./0123456789:;<??<
5?2
(?%
inputs?????????@@
p 

 
? "????????????
-__inference_sequential_layer_call_fn_35515224t'()*+,-./0123456789:;<??<
5?2
(?%
inputs?????????@@
p

 
? "????????????
&__inference_signature_wrapper_35514282?'()*+,-./0123456789:;<y?v
? 
o?l
4
input_1)?&
input_1?????????@@
4
input_2)?&
input_2?????????@@"/?,
*
lambda ?
lambda?????????