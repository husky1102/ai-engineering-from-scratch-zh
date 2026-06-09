(function () {
  var DEFAULT_LANG = 'zh-CN';
  var SUPPORTED = { en: true, 'zh-CN': true };
  var STORAGE_KEY = 'aifs.lang';

  var FALLBACKS = {
    en: {
      backToHome: 'Back to Home',
      chinese: '中文',
      close: 'Close',
      copied: 'Copied!',
      copy: 'Copy',
      copyCommand: 'Copy command',
      copyFailed: 'Copy failed',
      correctSuffix: 'correct',
      diagram: 'Diagram',
      diagramRenderError: 'Diagram could not be rendered.',
      english: 'English',
      languageLabel: 'Language',
      lessonNotFoundBody: 'Could not load the lesson at <code>{path}</code>. It may not have been written yet.',
      lessonNotFoundTitle: 'Lesson not found',
      loadingCodeFiles: 'Loading code files...',
      loadingLesson: 'Loading lesson...',
      loadingOutputs: 'Loading outputs...',
      midLessonCheck: 'Mid-Lesson Check',
      next: 'Next',
      noLessonBody: 'Add ?path=phases/01-math-foundations/01-linear-algebra-intuition to the URL.',
      noLessonTitle: 'No lesson path specified',
      onThisPage: 'On this page',
      postLessonQuiz: 'Post-Lesson Quiz',
      preLessonCheck: 'Pre-Lesson Check',
      previous: 'Previous',
      quiz: 'Quiz',
      renderErrorBody: 'Loaded the lesson markdown but failed to render it. Details are in the browser console.',
      renderErrorTitle: 'Render error',
      renderingDiagram: 'Rendering diagram...',
      runtimeLocal: 'Run locally',
      runtimeStatic: 'Static reading',
      runtimeTitle: 'Run the Code',
      runtimeLoading: 'Loading runtime...',
      runtimeNoCode: 'No executable code files detected for this lesson.',
      runtimeLocalNote: 'Use a local Python environment or notebook kernel for this lesson.',
      runtimeStaticNote: 'This lesson is best read as text or configuration.',
      runtimeNoteApiKey: 'Uses an API key or secret.',
      runtimeNoteEnvVars: 'Reads environment variables.',
      runtimeNoteSubprocess: 'Uses subprocess.',
      runtimeNoteNetwork: 'Uses network or server APIs.',
      runtimeNoteGpuCuda: 'Mentions GPU/CUDA.',
      runtimeNoteNoCodeFiles: 'No code files detected.',
      runtimeEntry: 'Entry',
      runtimePackages: 'Packages',
      runtimeNotes: 'Notes',
      openNotebook: 'Open notebook',
      notebookReady: 'Notebook generated for this lesson',
      openLab: 'Open Lab',
      outputsTitle: 'What This Lesson Ships',
      outputsSubtitle: 'Prompts, skills, and artifacts you can use right now',
      noOutputs: 'No output artifacts detected for this lesson.',
      openFile: 'Open file',
      install: 'Install',
      promptInstallHint: 'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts',
      sourceFiles: 'Source files',
      viewSource: 'View source',
      localAssetsMissing: 'Local lesson asset manifest is missing. Run scripts/build_lesson_assets.py.',
      learningPathTitle: 'Learning Path',
      phaseLabel: 'Phase',
      earlierLessons: 'earlier lessons',
      laterLessons: 'later lessons',
      phaseProgress: "You've completed {completed} of {total} lessons in this phase",
      readyForPhase: 'Ready for Phase {phase}: {name}',
      continueLearningTitle: 'Continue Learning',
      phaseFinished: 'You finished this phase!',
      browsePhaseLessons: 'Browse all Phase {phase} lessons',
      fullCourseCatalog: 'Full course catalog',
      personalPathCallout: 'Run /find-your-level in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed for a personalized learning path'
    },
    'zh-CN': {
      backToHome: '返回首页',
      chinese: '中文',
      close: '关闭',
      copied: '已复制',
      copy: '复制',
      copyCommand: '复制命令',
      copyFailed: '复制失败',
      correctSuffix: '题正确',
      diagram: '图示',
      diagramRenderError: '图示无法渲染。',
      english: 'English',
      languageLabel: '语言',
      lessonNotFoundBody: '无法加载 <code>{path}</code> 这节课。它可能还没有写好。',
      lessonNotFoundTitle: '未找到课程',
      loadingCodeFiles: '正在加载代码文件...',
      loadingLesson: '正在加载课程...',
      loadingOutputs: '正在加载交付物...',
      midLessonCheck: '课中检查',
      next: '下一课',
      noLessonBody: '请在 URL 中加入 ?path=phases/01-math-foundations/01-linear-algebra-intuition。',
      noLessonTitle: '缺少课程路径',
      onThisPage: '本页目录',
      postLessonQuiz: '课后测验',
      preLessonCheck: '课前检查',
      previous: '上一课',
      quiz: '测验',
      renderErrorBody: '课程 Markdown 已加载，但渲染失败。详情请查看浏览器控制台。',
      renderErrorTitle: '渲染错误',
      renderingDiagram: '正在渲染图示...',
      runtimeLocal: '本地运行',
      runtimeStatic: '静态阅读',
      runtimeTitle: '运行代码',
      runtimeLoading: '正在加载运行能力...',
      runtimeNoCode: '这节课没有检测到可执行代码文件。',
      runtimeLocalNote: '这节课需要本地 Python 环境或 notebook kernel。',
      runtimeStaticNote: '这节课更适合作为文本或配置阅读。',
      runtimeNoteApiKey: '使用 API key 或 secret。',
      runtimeNoteEnvVars: '读取环境变量。',
      runtimeNoteSubprocess: '使用 subprocess。',
      runtimeNoteNetwork: '使用网络或 server API。',
      runtimeNoteGpuCuda: '提到 GPU/CUDA。',
      runtimeNoteNoCodeFiles: '未检测到代码文件。',
      runtimeEntry: '入口',
      runtimePackages: '包',
      runtimeNotes: '说明',
      openNotebook: '打开 notebook',
      notebookReady: '这节课已生成 notebook',
      openLab: '打开 Lab',
      outputsTitle: '本课交付物',
      outputsSubtitle: '可直接复用的 prompts、skills 和产物',
      noOutputs: '这节课没有检测到交付物。',
      openFile: '打开文件',
      install: '安装',
      promptInstallHint: '复制到 Claude、Cursor、Codex、OpenClaw、Hermes 或任何读取 prompt 的 agent 中使用',
      sourceFiles: '源代码文件',
      viewSource: '查看源码',
      localAssetsMissing: '缺少本地 lesson 资产清单，请运行 scripts/build_lesson_assets.py。',
      learningPathTitle: '学习路径',
      phaseLabel: '阶段',
      earlierLessons: '前面的课程',
      laterLessons: '后面的课程',
      phaseProgress: '你已完成本阶段 {total} 课中的 {completed} 课',
      readyForPhase: '已准备进入第 {phase} 阶段：{name}',
      continueLearningTitle: '继续学习',
      phaseFinished: '你已完成本阶段！',
      browsePhaseLessons: '浏览第 {phase} 阶段全部课程',
      fullCourseCatalog: '完整课程目录',
      personalPathCallout: '在 Claude、Cursor、Codex、OpenClaw、Hermes 或任何安装了课程技能的智能体中运行 /find-your-level，生成个性化学习路径'
    }
  };

  function normalize(lang) {
    if (!lang) return '';
    var value = String(lang).trim();
    if (value.toLowerCase() === 'zh-cn' || value.toLowerCase() === 'zh') return 'zh-CN';
    if (value.toLowerCase() === 'en' || value.toLowerCase() === 'en-us') return 'en';
    return SUPPORTED[value] ? value : '';
  }

  function resolve(urlLang) {
    var lang = normalize(urlLang);
    if (!lang) {
      try { lang = normalize(localStorage.getItem(STORAGE_KEY)); } catch (_) {}
    }
    if (!lang) lang = DEFAULT_LANG;
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (_) {}
    return lang;
  }

  function merge(base, extra) {
    var out = {};
    Object.keys(base || {}).forEach(function (key) { out[key] = base[key]; });
    Object.keys(extra || {}).forEach(function (key) { out[key] = extra[key]; });
    return out;
  }

  function load(lang) {
    lang = normalize(lang) || DEFAULT_LANG;
    var fallback = merge(FALLBACKS.en, FALLBACKS[lang] || {});
    return fetch('i18n/' + lang + '.json')
      .then(function (res) {
        if (!res.ok) throw new Error('locale-fetch-failed');
        return res.json();
      })
      .then(function (dict) { return merge(fallback, dict); })
      .catch(function () {
        if (lang === 'en') return fallback;
        return fetch('i18n/en.json')
          .then(function (res) { return res.ok ? res.json() : {}; })
          .then(function (dict) { return merge(fallback, dict); })
          .catch(function () { return fallback; });
      });
  }

  function format(template, values) {
    return String(template || '').replace(/\{([a-zA-Z0-9_]+)\}/g, function (m, key) {
      return values && values[key] != null ? values[key] : m;
    });
  }

  window.AIFSI18N = {
    DEFAULT_LANG: DEFAULT_LANG,
    fallback: FALLBACKS,
    format: format,
    load: load,
    normalize: normalize,
    resolve: resolve,
    supported: SUPPORTED
  };
})();
