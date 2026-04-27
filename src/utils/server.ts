import { getCollection, type CollectionEntry } from 'astro:content'

export type BlogPostEntry = CollectionEntry<'blog'> | CollectionEntry<'blogEn'>

export const prod = import.meta.env.PROD

/** Strip optional /index(.md|.mdx) tail from a collection entry id. */
export function postFolder(id: string): string {
  return id.replace(/\/index(?:\.(?:md|mdx))?$/, '')
}

/** Canonical slug shared by both editions: drops trailing -zh. */
export function canonicalSlug(id: string): string {
  const folder = postFolder(id)
  return folder.endsWith('-zh') ? folder.slice(0, -3) : folder
}

/** A post is Chinese if its folder ends with -zh OR frontmatter language === 'zh'. */
function isZhPost(post: CollectionEntry<'blog'>): boolean {
  const folder = postFolder(post.id)
  if (folder.endsWith('-zh')) return true
  return post.data.language === 'zh'
}

async function loadAllBlogs() {
  return getCollection('blog', ({ data }) => (prod ? !data.draft : true))
}

/** Chinese collection only — used by the default `/blog/...` routes. */
export async function getBlogCollection() {
  const all = await loadAllBlogs()
  return all.filter(isZhPost)
}

/** English collection only, with fallback to Chinese-only orphans so nothing disappears
 *  from the EN listing if a translation is still missing. */
export async function getBlogCollectionEn() {
  const all = await loadAllBlogs()
  const enPosts = all.filter((p) => !isZhPost(p))
  const enSlugs = new Set(enPosts.map((p) => canonicalSlug(p.id)))
  const orphans = all.filter(isZhPost).filter((p) => !enSlugs.has(canonicalSlug(p.id)))
  return [...enPosts, ...orphans] as unknown as CollectionEntry<'blogEn'>[]
}

/** Return the counterpart slug if a sibling edition exists in the other language. */
export async function getCounterpartUrl(
  id: string,
  currentIsEn: boolean
): Promise<string | null> {
  const all = await loadAllBlogs()
  const slug = canonicalSlug(id)
  const want = (post: CollectionEntry<'blog'>) =>
    canonicalSlug(post.id) === slug && (currentIsEn ? isZhPost(post) : !isZhPost(post))
  const counterpart = all.find(want)
  if (!counterpart) return null
  const target = canonicalSlug(counterpart.id)
  return currentIsEn ? `/blog/${target}` : `/en/blog/${target}`
}

export async function getPostCollections() {
  return await getCollection('postCollections')
}

export async function getPostsForCollection(
  collection: CollectionEntry<'postCollections'>,
  isEn: boolean = false
) {
  const allPosts = isEn ? await getBlogCollectionEn() : await getBlogCollection()
  const blogList = collection.data.bloglist || []

  const postMap = new Map<string, BlogPostEntry>()
  allPosts.forEach((post) => {
    postMap.set(post.id.toLowerCase(), post)
    postMap.set(postFolder(post.id).toLowerCase(), post)
    postMap.set(canonicalSlug(post.id).toLowerCase(), post)
  })

  return blogList
    .map((itemId) => postMap.get(itemId.toLowerCase()))
    .filter((post): post is BlogPostEntry => post !== undefined)
}

function getYearFromCollection(collection: BlogPostEntry): number | undefined {
  const dateStr = collection.data.updatedDate ?? collection.data.publishDate
  return dateStr ? new Date(dateStr).getFullYear() : undefined
}

export function groupCollectionsByYear<T extends BlogPostEntry>(
  collections: T[]
): [number, T[]][] {
  const collectionsByYear = collections.reduce((acc, collection) => {
    const year = getYearFromCollection(collection)
    if (year !== undefined) {
      if (!acc.has(year)) {
        acc.set(year, [])
      }
      acc.get(year)!.push(collection)
    }
    return acc
  }, new Map<number, T[]>())

  return Array.from(collectionsByYear.entries()).sort((a, b) => b[0] - a[0])
}

export function sortMDByDate<T extends BlogPostEntry>(collections: T[]): T[] {
  return [...collections].sort((a, b) => {
    const aUpdatedDate = a.data.updatedDate ? new Date(a.data.updatedDate).valueOf() : 0
    const bUpdatedDate = b.data.updatedDate ? new Date(b.data.updatedDate).valueOf() : 0
    if (aUpdatedDate !== bUpdatedDate) {
      return bUpdatedDate - aUpdatedDate
    }
    const aPublishDate = a.data.publishDate ? new Date(a.data.publishDate).valueOf() : 0
    const bPublishDate = b.data.publishDate ? new Date(b.data.publishDate).valueOf() : 0
    return bPublishDate - aPublishDate
  })
}

export function getAllTags(collections: BlogPostEntry[]): string[] {
  return collections.flatMap((collection) => [...collection.data.tags])
}

export function getUniqueTags(collections: BlogPostEntry[]): string[] {
  return [...new Set(getAllTags(collections))]
}

export function getUniqueTagsWithCount(collections: BlogPostEntry[]): [string, number][] {
  return [
    ...getAllTags(collections).reduce(
      (acc, t) => acc.set(t, (acc.get(t) || 0) + 1),
      new Map<string, number>()
    )
  ].sort((a, b) => b[1] - a[1])
}

export function getAllCategories(collections: BlogPostEntry[]): string[] {
  return collections
    .map((collection) => collection.data.category)
    .filter((category): category is string => category !== undefined)
}

export function getUniqueCategories(collections: BlogPostEntry[]): string[] {
  return [...new Set(getAllCategories(collections))]
}

export function getUniqueCategoriesWithCount(
  collections: BlogPostEntry[]
): [string, number][] {
  return [
    ...getAllCategories(collections).reduce(
      (acc, c) => acc.set(c, (acc.get(c) || 0) + 1),
      new Map<string, number>()
    )
  ].sort((a, b) => b[1] - a[1])
}

export function getCollectionsByCategory<T extends BlogPostEntry>(
  collections: T[],
  category: string
): T[] {
  return collections.filter((collection) => collection.data.category === category)
}

export function getCollectionsByHie<T extends BlogPostEntry>(
  collections: T[],
  hie: string
): T[] {
  return collections.filter((collection) => (collection.data as any).hIE === hie)
}
