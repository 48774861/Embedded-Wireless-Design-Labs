///////////////////////////////////////////
//Add EvaluationKIT to Canvas//
///////////////////////////////////////////



/*EvaluationKIT START*/var evalkit_jshosted = document.createElement('script');evalkit_jshosted.setAttribute('defer', 'defer');evalkit_jshosted.setAttribute('type', 'text/javascript');evalkit_jshosted.setAttribute('src', 'https://smu.evaluationkit.com/canvas/js');document.getElementsByTagName('head')[0].appendChild(evalkit_jshosted);/*EvaluationKIT END*/



////////////////////////////////////////////////////
// DESIGNPLUS CONFIG                            //
////////////////////////////////////////////////////
// Legacy
var DT_variables = {
       iframeID: '',
        // Path to the hosted USU Design Tools
        path: 'https://designtools.ciditools.com/',
        templateCourse: '53347',
        // OPTIONAL: Button will be hidden from view until launched using shortcut keys
        hideButton: true,
    	 // OPTIONAL: Limit by course format
	     limitByFormat: false, // Change to true to limit by format
	     // adjust the formats as needed. Format must be set for the course and in this array for tools to load
	     formatArray: [
            'online',
            'on-campus',
            'blended'
        ],
        // OPTIONAL: Limit tools loading by users role
        limitByRole: false, // set to true to limit to roles in the roleArray
        // adjust roles as needed
        roleArray: [
            'student',
            'teacher',
            'admin'
        ],
        // OPTIONAL: Limit tools to an array of Canvas user IDs
        limitByUser: false, // Change to true to limit by user
        // add users to array (Canvas user ID not SIS user ID)
        userArray: [
            '1234',
            '987654'
        ] // Paste variables from existing code here
};

// New
DpPrimary = {
    lms: 'canvas',
    templateCourse: '134489',
    hideButton: true,
    hideLti: false,
    extendedCourse: '', // added in sub-account theme
    sharedCourse: '', // added from localStorage
    courseFormats: [],
    canvasRoles: [],
    canvasUsers: [],
    canvasCourseIds: [],
    plugins: [],
    excludedModules: [],
    includedModules: [],
    lang: 'en',
    defaultToLegacy: false,
    enableVersionSwitching: false,
    hideSwitching: false,
}

// merge with extended/shared customizations config
DpConfig = { ...DpPrimary, ...(window.DpConfig ?? {}) }

$(function () {
    const uriPrefix = (location.href.includes('.beta.')) ? 'beta.' : '';
    const toolsUri = (DpConfig.toolsUri) ? DpConfig.toolsUri : `https://${uriPrefix}designplus.ciditools.com/`;
    $.getScript(`${toolsUri}js/controller.js`);
});
////////////////////////////////////////////////////
// END DESIGNPLUS CONFIG                        //
////////////////////////////////////////////////////
// Start Pope Tech Accessibility Guide
var popeTechKey='33HYHp7k6YG6lVOAmPY95iTWwJRlmSWz';(function(a){function b(a,b){var c=document.createElement("script");c.type="text/javascript",c.readyState?c.onreadystatechange=function(){("loaded"===c.readyState||"complete"===c.readyState)&&(c.onreadystatechange=null,b())}:c.onload=function(){b()},c.src=a,document.getElementsByTagName("head")[0].appendChild(c)}function c(a){return a&&("TeacherEnrollment"===a||"TaEnrollment"===a||"DesignerEnrollment"===a)}function d(){var a=window.location.pathname;return!!(-1!==a.indexOf("/edit")||-1!==a.indexOf("/new")||-1!==a.indexOf("/syllabus")||a.match(/\/courses\/[0-9]+\/pages\/?$/)||a.match(/\/courses\/[0-9]+\/?$/))}function e(){return f()||g()}function f(){var a=/\/courses\/[0-9]+\/pages\/?$/,b=window.location.pathname;return console.log("Check for pages url",window.location.pathname),console.log(a.test(b)),a.test(b)}function g(){var a=window.location.pathname;return console.log("Check for courses url",window.location.pathname),console.log("/courses"===a),"/courses"===a}function h(){var a=/\/accounts\/[0-9]+\/external_tools\/[0-9]+\/?$/,b=/\/courses\/[0-9]+\/external_tools\/[0-9]+\/?$/,c=window.location.pathname;return console.log("Check for external tool url",window.location.pathname),console.log(a.test(c)||b.test(c)),a.test(c)||b.test(c)}function i(f){for(var g=0;g<f.length;++g)if(localStorage.setItem(`${j}.${l}`,JSON.stringify(f)),c(f[g].type)){if(d()&&b("https://canvas-cdn.pope.tech/loader.min.js",function(){}),null===a)break;(e()||h())&&(console.log("Load column"),b(`https://canvas-cdn.pope.tech/accessibility-column.min.js?key=${a}`,function(){}));break}}var j="pt-instructor-guide",k="username",l="enrollments";return-1===window.location.href.indexOf("/login/canvas")?-1===window.location.href.indexOf("?login_success=1")?void function(a,b){var c=localStorage.getItem(`${a}.${b}`);null===c?$.get("/api/v1/users/self/enrollments?type[]=DesignerEnrollment&type[]=TaEnrollment&type[]=TeacherEnrollment",function(a){i(a)}):(c=JSON.parse(c),i(c))}("pt-instructor-guide",l):(localStorage.removeItem(`${"pt-instructor-guide"}.${k}`),void $.get("/api/v1/users/self",function(a){localStorage.setItem(`${"pt-instructor-guide"}.${k}`,a.name)})):(localStorage.removeItem(`${"pt-instructor-guide"}.${k}`),localStorage.removeItem(`${"pt-instructor-guide"}.${"uuid"}`),localStorage.removeItem(`${"pt-instructor-guide"}.${"settings"}`),void localStorage.removeItem(`${"pt-instructor-guide"}.${l}`))})(popeTechKey);
// End Pope Tech Accessibility Guide