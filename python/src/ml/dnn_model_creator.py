from keras.engine import Model
from keras.layers import Dropout, GlobalMaxPooling1D, Dense, Conv1D, MaxPooling1D, Bidirectional, Concatenate
from keras.layers import LSTM
from keras.models import Sequential


def create_model_with_branch(embedding_layer, model_descriptor:str):
    "sub_conv[2,3,4](dropout=0.2,conv1d=100-v,)"
    submod_str_start=model_descriptor.index("sub_conv")
    submod_str_end=model_descriptor.index(")")
    submod_str=model_descriptor[submod_str_start: submod_str_end]

    kernel_str=submod_str[submod_str.index("[")+1: submod_str.index("]")]
    submod_layer_descriptor = submod_str[submod_str.index("(")+1:]
    submodels = []
    for ks in kernel_str.split(","):
        model = Sequential()
        model.add(embedding_layer)
        for layer_descriptor in submod_layer_descriptor.split(","):
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
                                 kernel_size=int(ks), padding='same', activation='relu'))
            elif layer_name=="maxpooling1d":
                size=params[0]
                if size=="v":
                    size=int(ks)
                else:
                    size=int(params[0])
                model.add(MaxPooling1D(pool_size=size))
            elif layer_name=="gmaxpooling1d":
                model.add(GlobalMaxPooling1D())
            elif layer_name=="dense":
                model.add(Dense(int(params[0]), activation=params[1]))
        submodels.append(model)

    submodel_outputs = [model.output for model in submodels]
    x = Concatenate(axis=1)(submodel_outputs)

    parallel_layers=Model(inputs=embedding_layer.input, outputs=x)
    print("submodel:")
    parallel_layers.summary()
    print("")

    outter_model_descriptor=model_descriptor[model_descriptor.index(")")+2:]
    big_model = Sequential()
    big_model.add(parallel_layers)
    for layer_descriptor in outter_model_descriptor.split(","):
        ld=layer_descriptor.split("=")

        layer_name=ld[0]
        params=None
        if len(ld)>1:
            params=ld[1].split("-")

        if layer_name=="dropout":
            big_model.add(Dropout(float(params[0])))
        elif layer_name=="lstm":
            big_model.add(LSTM(units=int(params[0]), return_sequences=bool(params[1])))
        elif layer_name=="bilstm":
            big_model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=bool(params[1]))))
        elif layer_name=="conv1d":
            big_model.add(Conv1D(filters=int(params[0]),
                                 kernel_size=int(params[1]), padding='same', activation='relu'))
        elif layer_name=="maxpooling1d":
            big_model.add(MaxPooling1D(pool_size=int(params[0])))
        elif layer_name=="gmaxpooling1d":
            big_model.add(GlobalMaxPooling1D())
        elif layer_name=="dense":
            big_model.add(Dense(int(params[0]), activation=params[1]))

    big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    big_model.summary()

    return big_model


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
