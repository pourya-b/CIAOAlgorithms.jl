# module ithello

using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators


struct Index_iterable{I<:Int}
  x0::I     # initial point of the sequence 
  m::I      # length of the sequence 
  d::I      # number of repetitions of each sequence ex) 123123456... has d = 2
  N::I      # max of index
end


mutable struct Index_state{I<:Int}
  i::I      # current index: ex: 123123123456...
  m_n::I    # memory index behind
  m_o::I    # memory index front
  m::I      # length of the sequence 
  cnt::I    # an inner counter
  x0::I     # an inner index
end


function Index_state(i::I, m_n::I, m_o::I, m::I) where {I}
    return Index_state{I}(
        i,
        m_n,
        m_o,
        m,
        0,
        1,

    )
end

function Base.iterate(iter::Index_iterable{I}) where {I}
    i =  (iter.x0 === nothing) ?  I(1) : iter.x0 

    state = Index_state(i, I(1), I(2), iter.m)
    return state, state
end

function Base.iterate(
        iter::Index_iterable,
        state::Index_state,
    )

    state.cnt += 1
        
    if mod(state.cnt, iter.d * state.m) == 0
        state.x0 += state.m 
        state.m_n = 0
        state.m_o = 1
        # the tail
        if state.x0 + state.m > iter.N
            state.m = iter.N - state.x0 + 1
            state.cnt = 0
            if state.m <= 0 
                return nothing
            end 
        end
    end 
    # println(state.m)
    t = state.i 
    state.i = mod(mod(t, iter.m), state.m) + state.x0

    t = state.m_n 
    state.m_n = mod(t, state.m+1) + 1

    t = state.m_o 
    state.m_o = mod(t, state.m+1) + 1
    


    return state, state
end



