//
// Created by demeuren on 02/07/18.
//

#ifndef SHAMAN_MEMORYSTORE_H
#define SHAMAN_MEMORYSTORE_H

#include <stack>
#include <vector>

template<typename errorType>
class MemoryStore
{
public:
    // used to store pointers to unused vectors
    thread_local static std::stack<std::vector<errorType>*> memoryStore;

    /*
     * returns some memory, taken from the stack if possible
     */
    static std::vector<errorType>* getVector()
    {
        if (memoryStore.empty())
        {
            // creates new vector
            return new std::vector<errorType>();
        }
        else
        {
            // gets an existing (empty) vector
            auto memory = memoryStore.top();
            memoryStore.pop();
            return memory;
        }
    }

    /*
     * stores the memory in the stack
     */
    static void releaseVector(std::vector<errorType>* memory)
    {
        // clear memory (while keeping capacity intact for later resizing)
        memory->clear();

        // stores memory
        memoryStore.push(memory);
    }
};

template<typename errorType>
thread_local std::stack<std::vector<errorType>*> MemoryStore<errorType>::memoryStore;

#endif //SHAMAN_MEMORYSTORE_H
