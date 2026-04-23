import type { CardListData, Config, IntegrationUserConfig, ThemeUserConfig } from './types'

export const theme: ThemeUserConfig = {
  // === Basic configuration ===
  /** Title for your website. Will be used in metadata and as browser tab title. */
  title: "CrepuscularIRIS Blog",
  /** Will be used in index page & copyright declaration */
  author: 'CrepuscularIRIS',
  author_en: 'CrepuscularIRIS',
  /** Description metadata for your website. Can be used in page metadata. */
  description: '工程圈子与科研圈子的技术博客',
  description_en: 'Tech Blog for Engineering and Research',
  /** The default favicon for your site which should be a path to an image in the `public/` directory. */
  favicon: '/favicon/favicon.ico',
  /** Specify the default language for this site. */
  locale: {
    lang: 'zh-CN',
    attrs: 'zh_CN',
    // Date locale
    dateLocale: 'zh-CN',
    dateOptions: {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    }
  },
  /** Set a logo image to show in the homepage. */
  logo: {
    src: 'src/assets/avatar.png',
    alt: 'Avatar'
  },

  // === Global configuration ===
  titleDelimiter: '•',
  prerender: true,
  npmCDN: 'https://cdn.jsdelivr.net/npm',

  // in test
  head: [],
  customCss: [],

  /** Configure the header of your site. */
  header: {
    menu: [
      { title: 'Engineering', link: '/blog/engineering' },
      { title: 'Research', link: '/blog/research' },
      { title: 'Interpretation', link: '/blog/interpretation' },
      { title: 'Activity', link: '/activity' },
      { title: 'Projects', link: '/projects' },
      { title: 'Links', link: '/links' },
      { title: 'About', link: '/about' }
    ]
  },

  /** Configure the footer of your site. */
  footer: {
    // Registration information for ICP (optional)
    registration: {
      // url: '',
      // text: '',
      // website: '' // only show ICP if url === website
    },
    /** Enable displaying a "Astro & Axi theme powered" link in your site's footer. */
    credits: true,
    /** Optional details about the social media accounts for this site. */
    social: { github: 'https://github.com/CrepuscularIRIS' }
  },

  content: {
    externalLinksContent: ' ↗',
    /** Blog page size for pagination (optional) */
    blogPageSize: 15,
    externalLinkArrow: true, // show external link arrow
    // Currently support weibo, x, bluesky
    share: ['weibo', 'x', 'bluesky']
  },

  /** Personal information configuration */
  personal: {
    /** Your location */
    location: 'China',
    /** Your GitHub username */
    githubUsername: 'CrepuscularIRIS',
    /** Your email address */
    email: 'blog@example.com',
    /** Your Google Scholar profile URL */
    googleScholar: '',
    /** Blog start date for statistics */
    blogStartDate: '2025-03-18',
    /** Domain configuration */
    domains: {
      main: 'localhost:4321',
      githubPages: 'CrepuscularIRIS.github.io',
      // cloudflare: '',
      // friendCircle: '',
    },
  }
}

export const integ: IntegrationUserConfig = {
  links: {
    logbook: [
    ],
    // Yourself link info
    applyTip: [
      { name: 'Name', val: theme.title },
      { name: 'Desc', val: theme.description || 'Null' },
      { name: 'Link', val: `https://${theme.personal?.domains?.main || 'example.com'}` },
      { name: 'Avatar', val: `https://${theme.personal?.domains?.main || 'example.com'}/avatar/avatar.png` }
    ]
  },
  // Enable page search function
  pagefind: true,
  // Add a random quote to the footer (default on homepage footer)
  quote: {
    // https://developer.hitokoto.cn/sentence/#%E8%AF%B7%E6%B1%82%E5C%B0%E5%9D%80
    // server: 'https://v1.hitokoto.cn/?c=i',
    // target: (data) => (data as { hitokoto: string }).hitokoto || 'Error'
    // https://github.com/lukePeavey/quotable
    server: 'https://api.quotable.io/quotes/random?maxLength=60',
    target: `(data) => data[0].content || 'Error'`
  },
  // Tailwindcss typography
  typography: {
    // https://github.com/tailwindlabs/tailwindcss-typography
    class:
      'break-words prose prose-axi dark:prose-invert dark:prose-axi prose-headings:font-medium'
  },
  // A lightbox library that can add zoom effect
  mediumZoom: {
    enable: true, // disable it will not load the whole library
    selector: '.prose .zoomable',
    options: {
      className: 'zoomable'
    }
  },
  // Comment system
  waline: {
    enable: false,
    // Server service link
    server: 'https://waline.example.com/',
    // Refer https://waline.js.org/en/guide/features/emoji.html
    emoji: ['bmoji', 'weibo'],
    // Refer https://waline.js.org/en/reference/client/props.html
    additionalConfigs: {
      // search: false,
      pageview: true,
      comment: true,
      locale: {
        reaction0: 'Like',
        placeholder: 'Welcome to comment. (Email to receive replies. Login is unnecessary)'
      },
      imageUploader: false
    }
  }
}

export const terms: CardListData = {
  title: 'Terms content',
  list: [
    {
      title: 'Privacy Policy',
      link: '/terms/privacy-policy'
    },
    {
      title: 'Terms and Conditions',
      link: '/terms/terms-and-conditions'
    },
    {
      title: 'Copyright',
      link: '/terms/copyright'
    },
    {
      title: 'Disclaimer',
      link: '/terms/disclaimer'
    }
  ]
}

const config = { ...theme, integ } as Config
export default config
