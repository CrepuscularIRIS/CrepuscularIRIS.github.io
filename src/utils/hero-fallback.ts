// Static fallback images per hIE category. Astro requires static imports for
// asset processing, so the candidates are listed up-front; the picker is
// deterministic so the same post always renders the same hero across reloads.

import kouka1 from '@/assets/hIE/kouka/181339-beatless-kouka-redjuice-thighhighs.jpg'
import kouka2 from '@/assets/hIE/kouka/188823-beatless-kouka-mecha-mecha_musume-redjuice-thighhighs-wallpaper.jpg'
import kouka3 from '@/assets/hIE/kouka/444647-animal_ears-beatless-cleavage-kouka-nyaa__28nnekoron_29-thighhighs.jpg'

import lacia1 from '@/assets/hIE/lacia/lacia-1920x1200-869498.jpg'
import lacia2 from '@/assets/hIE/lacia/lacia-futuristic-glow-909177.jpg'

import methode1 from '@/assets/hIE/methode/445955-beatless-bodysuit-dress-katou_hiromasa-lacia-methode-snowdrop.jpg'
import methode2 from '@/assets/hIE/methode/478880-beatless-bodysuit-fhilippedu-methode.jpg'
import methode3 from '@/assets/hIE/methode/669371-ass-beatless-bodysuit-dress-kouka-lacia-maid-mariage-mecha_musume-methode.jpg'

import snowdrop1 from '@/assets/hIE/snowdrop/291936-beatless-lacia-monochrome-redjuice-snowdrop.jpg'
import snowdrop2 from '@/assets/hIE/snowdrop/348914-beatless-no_bra-pointy_ears-redjuice-see_through-snowdrop-tattoo.jpg'

import shared1 from '@/assets/hIE/_shared/280186-multi-arato-kouka-lacia-methode-snowdrop.jpg'
import shared2 from '@/assets/hIE/_shared/cybernetic-922741.jpg'
import shared3 from '@/assets/hIE/_shared/group-5hIE-arato-alphacoders-896444.jpg'
import shared4 from '@/assets/hIE/_shared/group-kengo-lacia-922740.jpg'

type ImgImport = ImageMetadata

const POOLS: Record<string, ImgImport[]> = {
  kouka: [kouka1, kouka2, kouka3],
  lacia: [lacia1, lacia2],
  methode: [methode1, methode2, methode3],
  snowdrop: [snowdrop1, snowdrop2],
  // No saturnus assets yet — fall back to shared.
  saturnus: [shared1, shared3],
  _shared: [shared1, shared2, shared3, shared4]
}

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

/** Pick a stable fallback hero image based on hIE category and slug. */
export function pickHeroFallback(hIE: string | undefined, slug: string): ImgImport {
  const pool = (hIE && POOLS[hIE]) || POOLS._shared
  return pool[hashString(slug) % pool.length]
}
