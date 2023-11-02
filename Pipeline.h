#pragma once
#include "Layers/Layer.h"
#include "Layers/Conv.h"
#include "Layers/ReLu.h"
#include "Layers/Dense.h"
#include "Layers/Flatten.h"
#include "Layers/Norm.h"
#include "Layers/Sigmoid.h"

#include <utility>
#include <type_traits>



template <::std::size_t ... Is>
constexpr auto indexSequenceReverse(::std::index_sequence<Is...> const&)
-> decltype(::std::index_sequence<sizeof...(Is) - 1U - Is...>{});


template <::std::size_t N>
using makeIndexSequenceReverse
= decltype(indexSequenceReverse(::std::make_index_sequence<N>{}));


template <typename... Layers>
class Pipeline {
public:

	template <::std::size_t... Dims>
	auto forward(Tensor<Dims...>& input) {
		return proccessLayers(input, std::index_sequence_for<Layers...>(), [](auto&& layer, auto&& data) {
			return layer.Forward(std::forward<decltype(data)>(data));
		});
	}

	template <::std::size_t... Dims>
	auto forward(Tensor<Dims...>&& input) {
		return proccessLayers(input, std::index_sequence_for<Layers...>(), [](auto&& layer, auto&& data) {
			return layer.Forward(std::forward<decltype(data)>(data));
		});
	}


	template <::std::size_t... Dims>
	auto backward(Tensor<Dims...>& input) {
		return proccessLayers(input, makeIndexSequenceReverse<sizeof...(Layers)>(), [](auto&& layer, auto&& data) {
			return layer.Backward(std::forward<decltype(data)>(data));
		});
	}

	template <::std::size_t... Dims>
	auto backward(Tensor<Dims...>&& input) {
		return proccessLayers(input, makeIndexSequenceReverse<sizeof...(Layers)>(), [](auto&& layer, auto&& data) {
			return layer.Backward(std::forward<decltype(data)>(data));
		});
	}

private:
	
	template <::std::size_t... Dims, size_t Index, typename Func, size_t... Rest>
	auto proccessLayers(Tensor<Dims...>& input, std::index_sequence<Index, Rest...>, Func&& func) {
		// Forward the input through the current layer
		auto output = func(std::get<Index>(layers), input);

		// Continue the forward pass with the rest of the layers
		return proccessLayers(output, index_sequence<Rest...>(), func);
	}

	template <::std::size_t... Dims, typename Func>
	auto proccessLayers(Tensor<Dims...>& input, std::index_sequence<>, Func&& func) {
		// When there are no more layers, return the final output
		return input;
	}

private:
	std::tuple<Layers...> layers;
};

