//
// Created by Vaclav Volhejn on 14.06.2022.
//

#ifndef INFERENCE_RUNTIMES_RUNTIME_H
#define INFERENCE_RUNTIMES_RUNTIME_H


class Runtime {
public:
    virtual float getLatency() = 0;

    virtual ~Runtime() {}
};

#endif //INFERENCE_RUNTIMES_RUNTIME_H
