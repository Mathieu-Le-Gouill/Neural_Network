#pragma once
#include "Layers/Layer.h"
#include <utility>
#include <type_traits>

template <std::size_t ... Is>
constexpr auto indexSequenceReverse(std::index_sequence<Is...> const&)
-> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});


template <std::size_t N>
using makeIndexSequenceReverse
= decltype(indexSequenceReverse(std::make_index_sequence<N>{}));


template <typename... Layers>
class Pipeline {
public:
	template <size_t... Dims>
	auto forward(Tensor<Dims...>& input) {
		return forwardHelper(input, std::index_sequence_for<Layers...>());
	}

	template <size_t... Dims>
	auto forward(Tensor<Dims...>&& input) {
		return forwardHelper(input, std::index_sequence_for<Layers...>());
	}


	template <size_t... Dims>
	auto backward(Tensor<Dims...>& input) {
		return backwardHelper(input, makeIndexSequenceReverse<sizeof...(Layers)>());
	}

	template <size_t... Dims>
	auto backward(Tensor<Dims...>&& input) {
		return backwardHelper(input, makeIndexSequenceReverse<sizeof...(Layers)>());
	}

private:
	template <size_t... Dims, size_t Index, size_t... Rest>
	auto forwardHelper(Tensor<Dims...>& input, std::index_sequence<Index, Rest...> ) {
		// Forward the input through the current layer
		auto output = std::get<Index>(layers).Forward(input);

		// Continue the forward pass with the rest of the layers
		return forwardHelper(output, std::index_sequence<Rest...>());
	}

	template <size_t... Dims>
	auto forwardHelper(Tensor<Dims...>& input, std::index_sequence<>) {
		// When there are no more layers, return the final output
		return input;
	}

	
	template <size_t... Dims, size_t Index, size_t... Rest>
	auto backwardHelper(Tensor<Dims...>& input, std::index_sequence<Index, Rest...>) {
		// Forward the input through the current layer
		auto output = std::get<Index>(layers).Backward(input);

		// Continue the forward pass with the rest of the layers
		return backwardHelper(output, index_sequence<Rest...>());
	}

	template <size_t... Dims>
	auto backwardHelper(Tensor<Dims...>& input, std::index_sequence<>) {
		// When there are no more layers, return the final output
		return input;
	}

private:
	std::tuple<Layers...> layers;
};

