// Tells the page that opened this cabinet what happened in it (src/play/ui/Cabinet.jsx listens). Nothing else leaves
// the frame. Escape is passed on when the game is not holding the pointer, so one key steps back from the cabinet.
(function () {
  var say = function (type, data) { try { parent.postMessage(Object.assign({ citadel: 'cabinet', type: type }, data || {}), location.origin) } catch (e) {} }
  addEventListener('keydown', function (e) { if (e.code === 'Escape' && !document.pointerLockElement) say('esc') })
  window.citadelCabinet = { say: say }
  say('ready')
})()
