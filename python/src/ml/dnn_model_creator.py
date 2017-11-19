from keras.layers import Dropout, GlobalMaxPooling1D, Dense, Conv1D, MaxPooling1D, Merge, Bidirectional
from keras.layers import LSTM
from keras.models import Sequential


def create_model_without_branch(embedding_layer, model_descriptor:str):
    model = Sequential()
    model.add(embedding_layer)
    for layer_descriptor in model_descriptor.split(","):
        ld=layer_descriptor.split("=")

        layer_name=ld[0]
        params=None
        if len(ld)>1:
            params=ld[1].split("-")

        if layer_name=="dropout":
            model.add(Dropout(float(params[0])))
        elif layer_name=="lstm":
            model.add(LSTM(units=int(params[0]), return_sequences=bool(params[1])))
        elif layer_name=="bilstm":
            model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=bool(params[1]))))
        elif layer_name=="conv1d":
            model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(params[1]), padding='same', activation='relu'))
        elif layer_name=="maxpooling1d":
            model.add(MaxPooling1D(pool_size=int(params[0])))
        elif layer_name=="gmaxpooling1d":
            model.add(GlobalMaxPooling1D())
        elif layer_name=="dense":
            model.add(Dense(int(params[0]), activation=params[1]))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def create_lstm_type1(embedding_layer):#start from simple model
    return create_model_without_branch(embedding_layer, "dropout=0.2,lstm=100-True,gmaxpooling1d,"
                                                        "dropout=0.2,dense=2-softmax")


def create_model_conv_lstm_type1(embedding_layer):
    return create_model_without_branch(embedding_layer,
                                       "dropout=0.2,conv1d=100-4,maxpooling1d=4,"
                                       "lstm=100-True,gmaxpooling1d,dense=2-softmax")


def create_model_conv_lstm_multi_filter(embedding_layer):
    flts=100
    kernel_sizes=[2,3,4]
    submodels = []
    for kw in kernel_sizes:    # kernel sizes
        submodel = Sequential()
        submodel.add(embedding_layer)
        submodel.add(Dropout(0.2))
        submodel.add(Conv1D(filters=flts,
                            kernel_size=kw,
                            padding='same',
                            activation='relu'))
        submodel.add(MaxPooling1D(pool_size=4))
        submodels.append(submodel)

    big_model = Sequential()
    #concat = Concatenate(axis=1)
    #big_model.add(concat(submodels))
    big_model.add(Merge(submodels, mode="concat", concat_axis=1))
    big_model.add(LSTM(units=100, return_sequences=True))
    big_model.add(GlobalMaxPooling1D())
    #big_model.add(Dropout(0.2))
    big_model.add(Dense(2, activation='softmax'))
    big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #logger.info("New run started at {}\n{}".format(datetime.datetime.now(), big_model.summary()))
    return big_model
