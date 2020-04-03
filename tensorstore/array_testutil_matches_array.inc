#ifndef TENSORSTORE_ARRAY_TESTUTIL_H_
#error "Do not include this file directly, include array_testutil.h instead."
#endif

/// Defines additional MatchesArray overloads.

// [BEGIN GENERATED: generate_matches_array_overloads.py]

/// Returns a GMock matcher that matches a rank-2 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1>
ArrayMatcher MatchesArray(
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-3 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2>
ArrayMatcher MatchesArray(
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-4 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayMatcher MatchesArray(
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-5 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayMatcher MatchesArray(
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3][N4]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-6 array with zero origin.
///
/// This overload can be called with a braced list.
///
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayMatcher MatchesArray(const ::testing::Matcher<Element> (
    &element_matchers)[N0][N1][N2][N3][N4][N5]) {
  return MatchesArray<Element>(MakeArray(element_matchers));
}

/// Returns a GMock matcher that matches a rank-2 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1>
ArrayMatcher MatchesArray(
    span<const Index, 2> origin,
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-3 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2>
ArrayMatcher MatchesArray(
    span<const Index, 3> origin,
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-4 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3>
ArrayMatcher MatchesArray(
    span<const Index, 4> origin,
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-5 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4>
ArrayMatcher MatchesArray(
    span<const Index, 5> origin,
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3][N4]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-6 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5>
ArrayMatcher MatchesArray(span<const Index, 6> origin,
                          const ::testing::Matcher<Element> (
                              &element_matchers)[N0][N1][N2][N3][N4][N5]) {
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-2 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(
    const Index (&origin)[OriginRank],
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1]) {
  static_assert(OriginRank == 2, "Origin vector must have length 2.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-3 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2,
          std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(
    const Index (&origin)[OriginRank],
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2]) {
  static_assert(OriginRank == 3, "Origin vector must have length 3.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-4 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3,
          std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(
    const Index (&origin)[OriginRank],
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3]) {
  static_assert(OriginRank == 4, "Origin vector must have length 4.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-5 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(
    const Index (&origin)[OriginRank],
    const ::testing::Matcher<Element> (&element_matchers)[N0][N1][N2][N3][N4]) {
  static_assert(OriginRank == 5, "Origin vector must have length 5.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}

/// Returns a GMock matcher that matches a rank-6 array with the specified
/// origin.
///
/// This overload can be called with a braced list.
///
/// \param origin The expected origin vector of the array.
/// \param element_matchers The matchers for each element of the array.
template <typename Element, Index N0, Index N1, Index N2, Index N3, Index N4,
          Index N5, std::ptrdiff_t OriginRank>
ArrayMatcher MatchesArray(const Index (&origin)[OriginRank],
                          const ::testing::Matcher<Element> (
                              &element_matchers)[N0][N1][N2][N3][N4][N5]) {
  static_assert(OriginRank == 6, "Origin vector must have length 6.");
  return MatchesArray<Element>(MakeOffsetArray(origin, element_matchers));
}
// [END GENERATED: generate_matches_array_overloads.py]
