// Derived from searchtools.js in sphinx.
//
// :copyright: Copyright 2007-2021 by the Sphinx team, see AUTHORS.
// :license: BSD, see LICENSE for details.

import escapeHTML from "escape-html"

import { configuration } from "~/_"

import { SearchResultItem } from "./integrations/search/_"

interface SphinxSearchResult {
  docurl: string
  title: string
  anchor: string
  objectLabel: string | null
  synopsis: string | null
  score: number
}

const config = configuration()

/**
 * Returns a URL for a given path relative to the documentation root.
 *
 * @param path - Path relative to documentation root.
 * @returns The full URL.
 */
function getAbsoluteUrl(path: string): string {
  return `${config.base}/${path}`
}

let searchIndexLoaded: Promise<void> | undefined

/**
 * Loads a script by adding a `<script>` tag, which works even when
 * loaded with `file://`.
 *
 * @param source - Source script, relative to the documentation root.
 * @returns Promise that resolves when loading completes
 */
function addScript(source: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script")
    const url = getAbsoluteUrl(source)
    el.src = url
    el.addEventListener("load", () => resolve())
    el.addEventListener("error", () => {
      // eslint-disable-next-line no-console
      console.error(`Failed to load search data: ${url}`)
      reject()
    })
    document.body.appendChild(el)
  })
}

/**
 * Loads the search index and language data if not already loaded.
 *
 * @returns Promise that resolves when the data is loaded.
 */
function loadSearchIndex(): Promise<void> {
  if (searchIndexLoaded !== undefined) return searchIndexLoaded
  searchIndexLoaded = Promise.all([
    addScript("_static/language_data.js"),
    addScript("searchindex.js")
  ]).then(() => {
 return
                })
  return searchIndexLoaded
}

/**
 * Simple result scoring code.
 */
const Scorer = {
  // Implement the following function to further tweak the score for each result
  // The function takes a result array [filename, title, anchor, descr, score]
  // and returns the new score.
  /*
  score: function(result) {
    return result[4];
  },
  */

  // query matches the full name of an object
  objNameMatch: 11,
  // or matches in the last dotted part of the object name
  objPartialMatch: 6,
  // Additive scores depending on the priority of the object
  objPrio: {
    0: 15, // used to be importantResults
    1: 5, // used to be objectResults
    2: -5
  } as Record<number, number>, // used to be unimportantResults
  //  Used when the priority is not in the mapping.
  objPrioDefault: 0,

  // query found in title
  title: 15,
  partialTitle: 7,
  // query found in terms
  term: 5,
  partialTerm: 2
}

interface SearchIndex {
  docurls: string[]
  objects: Record<string, Record<string, [documentIndex: number, objectTypeIndex: number, priority: number, anchor: string, synopsis: string]>>
  objnames: Record<number | string, string[]>
  objtypes: Record<number | string, string>
  terms: Record<string, number | number[]>
  titles: string[]
  titleterms: Record<string, number | number[]>
}

// Search index data.
//
// The `Search.setIndex` method is called by searchindex.js when it
// loads.
declare global {
  // eslint-disable-next-line no-var
  var Search: { setIndex(index: SearchIndex): void }
}
let searchIndex: SearchIndex
window.Search = {
  setIndex: (index: SearchIndex) => {
    searchIndex = index
  }
}

const DEBUG_SCORES = false

/**
 * Finds matches for the specified object name.
 *
 * @param objectTerm - Primary object term to find.
 * @param otherterms - Other object terms.
 * @returns List of matching documents.
 */
function performObjectSearch(objectTerm: string, otherterms: string[]): SphinxSearchResult[] {
  const {docurls, objects, objnames, titles} = searchIndex

  const results: SphinxSearchResult[] = []
  for (const prefix in objects) {
    for (const name in objects[prefix]) {
      const fullname = (prefix ? `${prefix}.` : "") + name
      const fullnameLower = fullname.toLowerCase()
      if (fullnameLower.indexOf(objectTerm) > -1) {
        let score = 0
        const parts = fullnameLower.split(".")
        // check for different match types: exact matches of full name or
        // "last name" (i.e. last dotted part)
        if (fullnameLower === objectTerm || parts[parts.length - 1] === objectTerm) {
          score += Scorer.objNameMatch
          // matches in last name
        } else if (parts[parts.length - 1].indexOf(objectTerm) > -1) {
          score += Scorer.objPartialMatch
        }
        const match = objects[prefix][name]
        const objname = objnames[match[1]][2]
        const title = titles[match[0]]
        const synopsis = match[4]
        // If more than one term searched for, we require other words to be
        // found in the name/title/description
        if (otherterms.length > 0) {
          const haystack = `${prefix} ${name} ${objname} ${title} ${synopsis}`.toLowerCase()
          let allfound = true
          for (let i = 0; i < otherterms.length; i++) {
            if (haystack.indexOf(otherterms[i]) === -1) {
              allfound = false
              break
            }
          }
          if (!allfound) {
            continue
          }
        }

        let anchor = match[3]
        if (anchor === "") anchor = fullname
        else if (anchor === "-") anchor = `${objnames[match[1]][1]}-${fullname}`
        // add custom score for some objects according to scorer
        score += Scorer.objPrio[match[2]] ?? Scorer.objPrioDefault
        results.push({
          docurl: docurls[match[0]],
          title: fullname,
          anchor: `#${anchor}`,
          objectLabel: objname,
          synopsis,
          score
        })
      }
    }
  }

  return results
}

declare global {

  /**
   * Stemmer, defined by language_data.js.
   */
  class Stemmer {
    public stemWord(word: string): string
  }

  /**
   * Stopwords (search terms to ignore), defined by language_data.js
   */
  const stopwords: string[]
}

/**
 * Splits a search query into a list of terms.
 *
 * @param query - Raw search query.
 * @returns Split query terms.
 */
function splitQuery(query: string) {
  return query.split(/\s+/)
}

/**
 * Returns a regexp that matches a literal string.
 *
 * @param s - String to match.
 * @returns Escaped regular expression.
 */
function escapeRegExp(s: string) {
  return s.replace(/[.*+\-?^${}()|[\]\\]/g, "\\$&") // $& means the whole matched string
}

/**
 * Finds matching documents (not objects) in the search index.
 *
 * The search index must have already been loaded.
 *
 * @param searchterms - Stemmed search terms that must be present.
 * @param excluded - Stemmed search terms that must NOT be present.
 * @returns List of matching documents.
 */
function performTermsSearch(
  searchterms: string[],
  excluded: string[],
): SphinxSearchResult[] {
  const {docurls, titles, terms, titleterms} = searchIndex

  const fileMap: Record<number, string[]> = {}
  const scoreMap: Record<string, Record<string, number>> = {}
  const results = []

  // perform the search on the required terms
  for (let i = 0; i < searchterms.length; i++) {
    const word = searchterms[i]
    const files: number[] = []
    const matches = [
      { files: terms[word], score: Scorer.term },
      { files: titleterms[word], score: Scorer.title }
    ]
    // add support for partial matches
    if (word.length > 2) {
      const word_regex = escapeRegExp(word)
      if (!terms[word]) {
        for (const w in terms) {
          if (w.match(word_regex)) {
            matches.push({ files: terms[w], score: Scorer.partialTerm })
          }
        }
      }
      if (!titleterms[word]) {
        for (const w in titleterms) {
          if (w.match(word_regex)) {
            matches.push({ files: titleterms[w], score: Scorer.partialTitle })
          }
        }
      }
    }

    // no match but word was a required one
    if (matches.every(o => o.files === undefined)) {
      break
    }
    // found search word in contents
    matches.forEach(o => {
      let matchingFiles = o.files
      if (matchingFiles === undefined) {
        return
      }

      if (!Array.isArray(matchingFiles)) {
        matchingFiles = [matchingFiles]
      }
      files.push(...matchingFiles)

      // set score for the word in each file to Scorer.term
      for (let j = 0; j < matchingFiles.length; j++) {
        const file = matchingFiles[j]
        if (!(file in scoreMap)) {
          scoreMap[file] = {}
        }
        scoreMap[file][word] = o.score
      }
    })

    // create the mapping
    for (let j = 0; j < files.length; j++) {
      const file = files[j]
      if (file in fileMap && fileMap[file].indexOf(word) === -1) {
        fileMap[file].push(word)
      } else {
        fileMap[file] = [word]
      }
    }
  }

  // now check if the files don't contain excluded terms
  for (const file in fileMap) {
    const fileNum = parseInt(file, 10)
    let valid = true

    // check if all requirements are matched
    const filteredTermCount = // as search terms with length < 3 are discarded:
      // ignore
      searchterms.filter(term => term.length > 2).length
    if (
      fileMap[fileNum].length !== searchterms.length &&
      fileMap[fileNum].length !== filteredTermCount
    ) {
      continue
    }

    // ensure that none of the excluded terms is in the search result
    for (let i = 0; i < excluded.length; i++) {
      const termsFiles = terms[excluded[i]]
      if (Array.isArray(termsFiles) ? termsFiles.indexOf(fileNum) !== -1 :
        termsFiles === fileNum) {
        valid = false
        break
      }
      const titleFiles = titleterms[excluded[i]]
      if (Array.isArray(titleFiles) ? titleFiles.indexOf(fileNum) !== -1 :
        titleFiles === fileNum) {
        valid = false
        break
      }
    }

    // if we have still a valid result we can add it to the result list
    if (valid) {
      // select one (max) score for the file.
      // for better ranking, we should calculate ranking by using words
      // statistics like basic tf-idf...
      const score = Math.max(...fileMap[file].map(w => scoreMap[file][w]))
      results.push({
        docurl: docurls[file],
        title: titles[file],
        anchor: "",
        objectLabel: null,
        synopsis: null,
        score
      })
    }
  }
  return results
}

/**
 * Parsed section of a result document.
 */
interface Section {

  /**
   * Section title, or "" for the initial section before the first heading.
   */
  title: string

  /**
   * Anchor for the section (including the leading "#"), or "" for the initial section.
   */
  anchor: string

  /**
   * Full text of the section (excludes text in any sub-sections).
   */
  text: string
}

/**
 * Parses a raw HTML document into a list of sections.
 *
 * @param htmlString - Raw HTML document.
 * @returns List of sections.
 */
function htmlToSections(
  htmlString: string
): Section[] {
  const doc = new DOMParser().parseFromString(
    htmlString,
    "text/html"
  ) as HTMLDocument
  doc.querySelectorAll(".headerlink").forEach(headerLink => {
    headerLink.parentNode?.removeChild(headerLink)
  })
  const content = doc.querySelector("[role=main]")
  if (content === null) {
    // eslint-disable-next-line no-console
    console.warn(
      "Content block not found. Sphinx search tries to obtain it " +
        "via '[role=main]'. Could you check your theme or template."
    )
    return []
  }
  const headers = content.querySelectorAll("h1, h2, h3, h4, h5, h6")
  let prevElement: Element | undefined
  const sections: Section[] = []
  const range = doc.createRange()
  const addSection = (
    startElement: Element | undefined,
    endElement: Element | undefined
  ) => {
    if (startElement !== undefined) {
      range.setStartAfter(startElement)
    } else {
      range.setStartBefore(content)
    }
    if (endElement !== undefined) {
      range.setEndBefore(endElement)
    } else {
      range.setEndAfter(content)
    }
    const text = range.toString().trim()
    const title = startElement?.textContent?.trim()
    if (!title && !text) return
    const anchor = startElement !== undefined ? `#${startElement.id}` : ""
    sections.push({ title: title ?? "", anchor, text })
  }
  headers.forEach(element => {
    const id = element.id
    if (!id) return
    const startElement = prevElement
    prevElement = element
    addSection(startElement, element)
  })
  addSection(prevElement, undefined)
  return sections
}

interface SectionMatch {

  /**
   * Score of section match.  Higher means better match.  This is the
   * sum of `Scorer.partialTitle` or `Scorer.partialTerm` (as
   * applicable) for each search term.  For efficiency we don't stem
   * the document content, and therefore don't attempt to distinguish
   * between full and partial term matches.
   */
  score: number

  /**
   * Title of the section, or '' for the root section.
   */
  title: string

  /**
   * The snippet text containing the first term match within the
   * section.
   */
  snippet: string

  /**
   * Anchor of the section (including leading "#"), or '' for the root
   * section.
   */
  anchor: string

  /**
   * Search terms found in the section.
   */
  terms: Record<string, boolean>
}

/**
 * Extracts per-section matches and snippets from the raw HTML document text.
 *
 * @param htmlText - Raw text of full HTML document.
 * @param hlterms - List of unstemmed (raw) search terms to highlight.
 * @returns List of section matches.
 */
function getSectionMatches(
  htmlText: string,
  hlterms: string[]
): SectionMatch[]|undefined {
  const sections = htmlToSections(htmlText)
  const keywordPatterns = hlterms.map(
    s => new RegExp(escapeRegExp(s), "im")
  )
  const sectionMatches: {
    sectionIndex: number
    score: number
    snippetIndex: number
    terms: Record<string, boolean>
  }[] = []
  // Find all sections that match at least one search term.
  for (let sectionIndex = 0; sectionIndex < sections.length; ++sectionIndex) {
    const section = sections[sectionIndex]
    let score = 0
    let snippetIndex = Infinity
    const terms: Record<string, boolean> = {}
    for (let termIndex = 0, numTerms = hlterms.length; termIndex < numTerms; ++termIndex) {
      const pattern = keywordPatterns[termIndex]
      let m = section.title.match(pattern)
      let found = false
      if (m !== null) {
        snippetIndex = 0
        score += Scorer.partialTitle
        found = true
      } else {
        m = section.text.match(pattern)
        if (m !== null) {
          score += Scorer.partialTerm
          snippetIndex = Math.min(snippetIndex, m.index!)
          found = true
        }
      }
      terms[hlterms[termIndex]] = found
    }
    if (score !== 0) {
      sectionMatches.push({ sectionIndex, score, snippetIndex, terms })
    }
  }
  // Sort sections by score.
  sectionMatches.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score
    return a.sectionIndex - b.sectionIndex
  })
  if (sectionMatches.length === 0) {
    // Somehow no matches were found.
    return undefined
  }
  return sectionMatches.map(m => {
    const snippetLength = 240
    const section = sections[m.sectionIndex]
    const start = Math.max(m.snippetIndex - snippetLength / 2, 0)
    const excerpt =
      (start > 0 ? "\u2026" : "") +
      section.text.substr(start, snippetLength).trim() +
      (start + snippetLength < section.text.length ? "\u2026" : "")
    return {
      snippet: excerpt,
      anchor: section.anchor,
      title: section.title,
      score: m.score,
      terms: m.terms
    }
  })
}

/**
 * Converts a single sphinx search result to a mkdocs-material search
 * result including section and snippet information.
 *
 * @param result - The result obtained from the sphinx index.
 * @param hlterms - List of unstemmed (raw) search terms.
 * @param highlight - Function that adds search term highlighting markup
 * to a text string.
 * @returns Converted result.
 */
async function convertSphinxResult(
  result: SphinxSearchResult,
  hlterms: string[],
  highlight: (s: string) => string
): Promise<SearchResultItem> {
  const location = getAbsoluteUrl(result.docurl) + result.anchor
  // The title provided by the sphinx search index can include HTML
  // markup, which we need to strip.
  const title = stripHTML(result.title)
  const allTerms: Record<string, boolean> = {}
  for (const term of hlterms) {
    allTerms[term] = true
  }
  if (result.objectLabel !== null) {
    return [{
      location,
      score: result.score,
      terms: allTerms,
      title: `${highlight(title)
      }<span class="search-result-objlabel">${
        escapeHTML(result.objectLabel)}</span>`,
      text: highlight(result.synopsis!)
    }]
  }

  // Text match: attempt to obtain section and snippet information.
  const requestUrl = getAbsoluteUrl(result.docurl)

  let sectionMatches: SectionMatch[]|undefined
  // `fetch` is not supported for file:// URLs
  if (window.location.protocol !== "file:") {
    try {
      const resp = await fetch(requestUrl)
      const rawData = await resp.text()
      sectionMatches = getSectionMatches(rawData, hlterms)
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("Failed to retrieve search result document: ", e)
    }
  }
  if (sectionMatches === undefined) {
    sectionMatches = [{score: -1, title: "", anchor: "", snippet: ""}]
  }
  // Add entry for parent document.
  const searchResults: SearchResultItem = []
  if (sectionMatches[0].score !== -1) {
    searchResults.push({
        location,
        score: result.score,
        terms: allTerms,
        title: highlight(title),
        text: ""
      })
  }
  let firstScore: number | undefined
  for (const m of sectionMatches) {
    if (firstScore === undefined) firstScore = m.score
    searchResults.push({
      location: location + m.anchor,
      // Give lower score to worse matches so that they are shown as
      // "more results".
      score: m.score === firstScore ? result.score : 0,
      terms: m.terms,
      title: highlight(m.title || title),
      text: highlight(m.snippet)
    })
  }
  return searchResults
}

export interface SearchResultStream {

  /**
   * Number of results.
   */
  count: number

  /**
   * Retrieves an individual search result.
   *
   * @param index - Result index, in range `[0, count)`.
   */
  get(index: number): Promise<SearchResultItem>
}

/**
 * Strips HTML tags and returns just the text content.
 *
 * @param html - The html document/fragment.
 * @returns The stripped text.
 */
function stripHTML(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html")
  return doc.body.textContent || ""
}

/**
 * Returns the search results for the given query.
 *
 * @param query - The raw search query.
 * @returns Search result stream.
 */
export async function getResults(query: string): Promise<SearchResultStream> {
  await loadSearchIndex()
  // stem the searchterms and add them to the correct list
  const stemmer = new Stemmer()

  // Stemmed and lowercased search terms that must be present.
  const searchterms: string[] = []

  // Stemmed and lowercased search terms that must not be present.
  const excluded: string[] = []

  // Unstemmed, lowercased search terms to highlight.
  const hlterms: string[] = []

  // Object search terms.
  const objectterms = []

  for (const origTerm of splitQuery(query)) {
    const lowerTerm = origTerm.toLowerCase()
    if (lowerTerm.length === 0) {
      continue
    }
    objectterms.push(lowerTerm)

    if (stopwords.indexOf(lowerTerm) !== -1) {
      // skip this "word"
      continue
    }
    // stem the word
    let word = stemmer.stemWord(lowerTerm)
    // prevent stemmer from cutting word smaller than two chars
    if (word.length < 3 && lowerTerm.length >= 3) {
      word = lowerTerm
    }
    let toAppend: string[]
    // select the correct list
    if (word[0] === "-") {
      toAppend = excluded
      word = word.substr(1)
    } else {
      toAppend = searchterms
      hlterms.push(lowerTerm)
    }
    // only add if not already in the list
    if (toAppend.indexOf(word) === -1) {
      toAppend.push(word)
    }
  }

  // console.debug('SEARCH: searching for:');
  // console.info('required: ', searchterms);
  // console.info('excluded: ', excluded);

  const results: SphinxSearchResult[] = []

  // lookup as object
  for (let i = 0; i < objectterms.length; i++) {
    const others = [
      ...objectterms.slice(0, i),
      ...objectterms.slice(i + 1, objectterms.length)
    ]
    results.push(...performObjectSearch(objectterms[i], others))
  }

  // lookup as search terms in fulltext
  results.push(...performTermsSearch(searchterms, excluded))
  // let the scorer override scores with a custom scoring function
  // FIXME:
  // if (Scorer.score) {
  //   for (i = 0; i < results.length; i++)
  //     results[i][4] = Scorer.score(results[i]);
  // }

  // now sort the results by score and then alphabetically
  results.sort((a, b) => {
    const left = a.score
    const right = b.score
    if (left !== right) return right - left
    // same score: sort alphabetically
    const leftTitle = a.title.toLowerCase()
    const rightTitle = b.title.toLowerCase()
    return leftTitle > rightTitle ? 1 : leftTitle < rightTitle ? -1 : 0
  })
  if (DEBUG_SCORES) {
    // eslint-disable-next-line no-console
    console.log(results)
  }

  const pattern = new RegExp(
    `\\b(?:${  hlterms.map(escapeRegExp).join("|")  })`,
    "img"
  )
  const highlight = (s: unknown) => {
    return `<mark data-md-highlight>${s}</mark>`
  }
  const highlightTerms = (text: string) => {
    return escapeHTML(text)
      .replace(pattern, highlight)
      .replace(/<\/mark>(\s+)<mark[^>]*>/gim, "$1")
  }

  return {
    count: results.length,
    get: (index: number) => {
      return convertSphinxResult(results[index], hlterms, highlightTerms)
    }
  }
}
