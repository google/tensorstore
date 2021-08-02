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
  animationFrameScheduler,
  interval,
  of
} from "rxjs"
import {
  concatMap,
  debounce,
  distinctUntilKeyChanged, observeOn
} from "rxjs/operators"

import {
  addToSearchResultList,
  resetSearchResultList,
  resetSearchResultMeta,
  setSearchResultMeta
} from "~/actions"
import {
  getElementOrThrow
} from "~/browser"
import {
  SearchResult
} from "~/integrations"
import { SearchResultStream, getResults } from "~/sphinx_search"
import { renderSearchResultItem } from "~/templates"

import { Component } from "../../_"
import { SearchQuery } from "../query"

/* ----------------------------------------------------------------------------
 * Helper types
 * ------------------------------------------------------------------------- */

/**
 * Mount options
 */
interface MountOptions {
  query$: Observable<SearchQuery>      /* Search query observable */
}

/* ----------------------------------------------------------------------------
 * Functions
 * ------------------------------------------------------------------------- */

/**
 * Mount search result list
 *
 * This function performs a lazy rendering of the search results, depending on
 * the vertical offset of the search result container.
 *
 * @param el - Search result list element
 * @param options - Options
 *
 * @returns Search result list component observable
 */
export function mountSearchResult(
  el: HTMLElement, { query$ }: MountOptions
): Observable<Component<SearchResult>> {
  /* Retrieve nested components */
  const meta = getElementOrThrow(":scope > :first-child", el)
  const list = getElementOrThrow(":scope > :last-child", el)

  let lastResults: SearchResultStream|undefined
  let blocked: (() => void)|undefined

  const scrollContainer = el.parentElement!
  const threshold = 16
  const atScrollLimit = () =>
      scrollContainer.scrollTop + scrollContainer.clientHeight + threshold >
    scrollContainer.scrollHeight
  const checkScrollLimit = () => {
    if (blocked === undefined) return
    if (atScrollLimit()) {
      blocked()
      blocked = undefined
    }
  }
  scrollContainer.addEventListener("scroll", checkScrollLimit, {passive: true})
  window.addEventListener("resize", checkScrollLimit, {passive: true})
  const startAddingResults = async (results: SearchResultStream) => {
    lastResults = results
    const blockSize = 4
    let limit = blockSize
    for (let i = 0; i < results.count; ++i) {
      if (i === limit) {
        if (!atScrollLimit()) {
          await new Promise(resolve => {
            blocked = () => resolve(undefined)
          })
        }
        limit += blockSize
      }
      if (lastResults !== results) {
        // Cancelled.
        return
      }
      const result = await results.get(i)
      if (lastResults !== results) {
        // Cancelled.
        return
      }
      addToSearchResultList(list, renderSearchResultItem(result))
    }
  }
  query$
      .pipe(
        distinctUntilKeyChanged("value"), debounce(() => interval(250)),
        concatMap(async query => {
            if (!query.value) return undefined
            return getResults(query.value)
          }),
        observeOn(animationFrameScheduler))
      .subscribe(results => {
        resetSearchResultList(list)
        if (results) {
          setSearchResultMeta(meta, results.count)
          void startAddingResults(results)
        } else {
          resetSearchResultMeta(meta)
        }
      })
  return of()
}
