/*
 * Copyright (c) 2016-2021 Martin Donath <martin.donath@squidfunk.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

import {
  Observable,
  of
} from "rxjs"
import {
  filter
} from "rxjs/operators"

import { Keyboard } from "~/browser"

import { Component, getComponentElement } from "../../_"

/* ----------------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------------- */

/**
 * Search suggestions
 */
export interface SearchSuggest {}

/* ----------------------------------------------------------------------------
 * Helper types
 * ------------------------------------------------------------------------- */

/**
 * Mount options
 */
interface MountOptions {
  keyboard$: Observable<Keyboard>      /* Keyboard observable */
}

/* ----------------------------------------------------------------------------
 * Functions
 * ------------------------------------------------------------------------- */

/**
 * Mount search suggestions
 *
 * This function will perform a lazy rendering of the search results, depending
 * on the vertical offset of the search result container.
 *
 * @param el - Search result list element
 * @param options - Options
 *
 * @returns Search result list component observable
 */
export function mountSearchSuggest(
  el: HTMLElement, { keyboard$ }: MountOptions
): Observable<Component<SearchSuggest>> {

  /* Retrieve query component and track all changes */
  const query  = getComponentElement("search-query")

  /* Set up search keyboard handlers */
  keyboard$
    .pipe(
      filter(({ mode }) => mode === "search")
    )
      .subscribe(key => {
        switch (key.type) {

          /* Right arrow: accept current suggestion */
          case "ArrowRight":
            if (
              el.innerText.length &&
              query.selectionStart === query.value.length
            )
              query.value = el.innerText
            break
        }
      })

  /* Create and return component */
  return of()
}
