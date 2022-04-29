#ifndef TENSORSTORE_ARRAY_H_
#error "Do not include this file directly, include array.h instead."
#endif

// IWYU pragma: private, include "third_party/tensorstore/array.h"

/// Defines additional MakeArray, MakeArrayView, MakeOffsetArray and
/// MakeOffsetArrayView overloads.

// [BEGIN GENERATED: generate_make_array_overloads.py]

template <typename Element, Index N0, Index N1, Index N2>
ArrayView<Element, 3> MakeArrayView(Element (&array)[N0][N1][N2]) {
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3> layout(shape, byte_strides);
  return {&array[0][0][0], layout};
}
template <typename Element, Index N0, Index N1, Index N2>
ArrayView<const Element, 3> MakeArrayView(const Element (&array)[N0][N1][N2]) {
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3> layout(shape, byte_strides);
  return {&array[0][0][0], layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayView<Element, 4> MakeArrayView(Element (&array)[N0][N1][N2][N3]) {
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4> layout(shape, byte_strides);
  return {&array[0][0][0][0], layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayView<const Element, 4> MakeArrayView(
    const Element (&array)[N0][N1][N2][N3]) {
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4> layout(shape, byte_strides);
  return {&array[0][0][0][0], layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayView<Element, 5> MakeArrayView(Element (&array)[N0][N1][N2][N3][N4]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5> layout(shape, byte_strides);
  return {&array[0][0][0][0][0], layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayView<const Element, 5> MakeArrayView(
    const Element (&array)[N0][N1][N2][N3][N4]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5> layout(shape, byte_strides);
  return {&array[0][0][0][0][0], layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayView<Element, 6> MakeArrayView(Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6> layout(shape, byte_strides);
  return {&array[0][0][0][0][0][0], layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayView<const Element, 6> MakeArrayView(
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6> layout(shape, byte_strides);
  return {&array[0][0][0][0][0][0], layout};
}

template <typename Element, Index N0, Index N1, Index N2>
SharedArray<Element, 3> MakeArray(Element (&array)[N0][N1][N2]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1, Index N2>
SharedArray<Element, 3> MakeArray(const Element (&array)[N0][N1][N2]) {
  return MakeCopy(MakeArrayView(array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3>
SharedArray<Element, 4> MakeArray(Element (&array)[N0][N1][N2][N3]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3>
SharedArray<Element, 4> MakeArray(const Element (&array)[N0][N1][N2][N3]) {
  return MakeCopy(MakeArrayView(array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
SharedArray<Element, 5> MakeArray(Element (&array)[N0][N1][N2][N3][N4]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
SharedArray<Element, 5> MakeArray(const Element (&array)[N0][N1][N2][N3][N4]) {
  return MakeCopy(MakeArrayView(array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
SharedArray<Element, 6> MakeArray(Element (&array)[N0][N1][N2][N3][N4][N5]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
SharedArray<Element, 6> MakeArray(
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  return MakeCopy(MakeArrayView(array));
}

template <typename Element, Index N0, Index N1, Index N2>
ArrayView<Element, 3, offset_origin> MakeOffsetArrayView(
    span<const Index, 3> origin, Element (&array)[N0][N1][N2]) {
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2>
ArrayView<const Element, 3, offset_origin> MakeOffsetArrayView(
    span<const Index, 3> origin, const Element (&array)[N0][N1][N2]) {
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayView<Element, 4, offset_origin> MakeOffsetArrayView(
    span<const Index, 4> origin, Element (&array)[N0][N1][N2][N3]) {
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayView<const Element, 4, offset_origin> MakeOffsetArrayView(
    span<const Index, 4> origin, const Element (&array)[N0][N1][N2][N3]) {
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayView<Element, 5, offset_origin> MakeOffsetArrayView(
    span<const Index, 5> origin, Element (&array)[N0][N1][N2][N3][N4]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayView<const Element, 5, offset_origin> MakeOffsetArrayView(
    span<const Index, 5> origin, const Element (&array)[N0][N1][N2][N3][N4]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayView<Element, 6, offset_origin> MakeOffsetArrayView(
    span<const Index, 6> origin, Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayView<const Element, 6, offset_origin> MakeOffsetArrayView(
    span<const Index, 6> origin,
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2>
SharedArray<Element, 3, offset_origin> MakeOffsetArray(
    span<const Index, 3> origin, Element (&array)[N0][N1][N2]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2>
SharedArray<Element, 3, offset_origin> MakeOffsetArray(
    span<const Index, 3> origin, const Element (&array)[N0][N1][N2]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3>
SharedArray<Element, 4, offset_origin> MakeOffsetArray(
    span<const Index, 4> origin, Element (&array)[N0][N1][N2][N3]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3>
SharedArray<Element, 4, offset_origin> MakeOffsetArray(
    span<const Index, 4> origin, const Element (&array)[N0][N1][N2][N3]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
SharedArray<Element, 5, offset_origin> MakeOffsetArray(
    span<const Index, 5> origin, Element (&array)[N0][N1][N2][N3][N4]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
SharedArray<Element, 5, offset_origin> MakeOffsetArray(
    span<const Index, 5> origin, const Element (&array)[N0][N1][N2][N3][N4]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
SharedArray<Element, 6, offset_origin> MakeOffsetArray(
    span<const Index, 6> origin, Element (&array)[N0][N1][N2][N3][N4][N5]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
SharedArray<Element, 6, offset_origin> MakeOffsetArray(
    span<const Index, 6> origin,
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, ptrdiff_t OriginRank>
ArrayView<Element, 3, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2]) {
  static_assert(OriginRank == 3, "Origin vector must have length 3.");
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, ptrdiff_t OriginRank>
ArrayView<const Element, 3, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1][N2]) {
  static_assert(OriginRank == 3, "Origin vector must have length 3.");
  static constexpr Index shape[] = {N0, N1, N2};
  static constexpr Index byte_strides[] = {
      N1 * N2 * sizeof(Element), N2 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<3, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3,
          ptrdiff_t OriginRank>
ArrayView<Element, 4, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2][N3]) {
  static_assert(OriginRank == 4, "Origin vector must have length 4.");
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3,
          ptrdiff_t OriginRank>
ArrayView<const Element, 4, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1][N2][N3]) {
  static_assert(OriginRank == 4, "Origin vector must have length 4.");
  static constexpr Index shape[] = {N0, N1, N2, N3};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * sizeof(Element), N2 * N3 * sizeof(Element),
      N3 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<4, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          ptrdiff_t OriginRank>
ArrayView<Element, 5, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2][N3][N4]) {
  static_assert(OriginRank == 5, "Origin vector must have length 5.");
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          ptrdiff_t OriginRank>
ArrayView<const Element, 5, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank],
    const Element (&array)[N0][N1][N2][N3][N4]) {
  static_assert(OriginRank == 5, "Origin vector must have length 5.");
  static constexpr Index shape[] = {N0, N1, N2, N3, N4};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * sizeof(Element), N2 * N3 * N4 * sizeof(Element),
      N3 * N4 * sizeof(Element), N4 * sizeof(Element), sizeof(Element)};
  StridedLayoutView<5, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5, ptrdiff_t OriginRank>
ArrayView<Element, 6, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank],
    Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static_assert(OriginRank == 6, "Origin vector must have length 6.");
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5, ptrdiff_t OriginRank>
ArrayView<const Element, 6, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank],
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static_assert(OriginRank == 6, "Origin vector must have length 6.");
  static constexpr Index shape[] = {N0, N1, N2, N3, N4, N5};
  static constexpr Index byte_strides[] = {
      N1 * N2 * N3 * N4 * N5 * sizeof(Element),
      N2 * N3 * N4 * N5 * sizeof(Element),
      N3 * N4 * N5 * sizeof(Element),
      N4 * N5 * sizeof(Element),
      N5 * sizeof(Element),
      sizeof(Element)};
  StridedLayoutView<6, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0][0][0][0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, Index N2, ptrdiff_t OriginRank>
SharedArray<Element, 3, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2]) {
  static_assert(OriginRank == 3, "Origin vector must have length 3.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, ptrdiff_t OriginRank>
SharedArray<Element, 3, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1][N2]) {
  static_assert(OriginRank == 3, "Origin vector must have length 3.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3,
          ptrdiff_t OriginRank>
SharedArray<Element, 4, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2][N3]) {
  static_assert(OriginRank == 4, "Origin vector must have length 4.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3,
          ptrdiff_t OriginRank>
SharedArray<Element, 4, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1][N2][N3]) {
  static_assert(OriginRank == 4, "Origin vector must have length 4.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          ptrdiff_t OriginRank>
SharedArray<Element, 5, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1][N2][N3][N4]) {
  static_assert(OriginRank == 5, "Origin vector must have length 5.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          ptrdiff_t OriginRank>
SharedArray<Element, 5, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank],
    const Element (&array)[N0][N1][N2][N3][N4]) {
  static_assert(OriginRank == 5, "Origin vector must have length 5.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5, ptrdiff_t OriginRank>
SharedArray<Element, 6, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank],
    Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static_assert(OriginRank == 6, "Origin vector must have length 6.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5, ptrdiff_t OriginRank>
SharedArray<Element, 6, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank],
    const Element (&array)[N0][N1][N2][N3][N4][N5]) {
  static_assert(OriginRank == 6, "Origin vector must have length 6.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
// [END GENERATED: generate_make_array_overloads.py]
