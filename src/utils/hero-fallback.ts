// One-image-per-hIE fallback. The blog now uses a single user-drawn image per
// 5-hIE category (image.png in each src/assets/hIE/<category>/ dir). The pool
// model from the earlier 14-asset library is collapsed: every hIE category
// resolves to its single signature image, and posts without an explicit
// heroImage frontmatter render the matching category illustration.

import kouka from '@/assets/hIE/kouka/image.png'
import lacia from '@/assets/hIE/lacia/image.png'
import methode from '@/assets/hIE/methode/image.png'
import saturnus from '@/assets/hIE/saturnus/image.png'
import snowdrop from '@/assets/hIE/snowdrop/image.png'

import shared1 from '@/assets/hIE/_shared/280186-multi-arato-kouka-lacia-methode-snowdrop.jpg'
import shared2 from '@/assets/hIE/_shared/cybernetic-922741.jpg'
import shared3 from '@/assets/hIE/_shared/group-5hIE-arato-alphacoders-896444.jpg'
import shared4 from '@/assets/hIE/_shared/group-kengo-lacia-922740.jpg'

type ImgImport = ImageMetadata

const HIE_IMAGE: Record<string, ImgImport> = {
  kouka,
  lacia,
  methode,
  saturnus,
  snowdrop
}

const SHARED_POOL: ImgImport[] = [shared1, shared2, shared3, shared4]

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

/** Pick a deterministic fallback hero. If hIE matches a known category, use
 *  that category's signature image. Otherwise fall back to the shared pool
 *  (used for legacy posts without an hIE field). */
export function pickHeroFallback(hIE: string | undefined, slug: string): ImgImport {
  if (hIE && HIE_IMAGE[hIE]) return HIE_IMAGE[hIE]
  return SHARED_POOL[hashString(slug) % SHARED_POOL.length]
}

/** Direct lookup by hIE category — used when migrating frontmatter refs. */
export function heroForHIE(hIE: string): ImgImport | undefined {
  return HIE_IMAGE[hIE]
}
