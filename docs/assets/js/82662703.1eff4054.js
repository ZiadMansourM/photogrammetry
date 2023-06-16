"use strict";(self.webpackChunkscanmate=self.webpackChunkscanmate||[]).push([[526],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>y});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),s=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=s(e.components);return r.createElement(l.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,l=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),u=s(n),m=a,y=u["".concat(l,".").concat(m)]||u[m]||d[m]||i;return n?r.createElement(y,o(o({ref:t},p),{},{components:n})):r.createElement(y,o({ref:t},p))}));function y(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=m;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c[u]="string"==typeof e?e:a,o[1]=c;for(var s=2;s<i;s++)o[s]=n[s];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},7934:(e,t,n)=>{n.r(t),n.d(t,{CenterContainer:()=>p,Table:()=>u,assets:()=>l,contentTitle:()=>o,default:()=>y,frontMatter:()=>i,metadata:()=>c,toc:()=>s});var r=n(7462),a=(n(7294),n(3905));const i={sidebar_position:1,id:"Introduction",description:"\ud83d\ude81 Helicopter view of Our Graduation Project.",slug:"/"},o=void 0,c={unversionedId:"Introduction",id:"Introduction",title:"Introduction",description:"\ud83d\ude81 Helicopter view of Our Graduation Project.",source:"@site/docs/ScanMate.md",sourceDirName:".",slug:"/",permalink:"/",draft:!1,editUrl:"https://github.com/ZiadMansourM/photogrammetry/tree/main/docs/ScanMate.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1,id:"Introduction",description:"\ud83d\ude81 Helicopter view of Our Graduation Project.",slug:"/"},sidebar:"tutorialSidebar",next:{title:"Under The Hood",permalink:"/category/under-the-hood"}},l={},s=[{value:"\ud83d\udcdd Pipeline",id:"-pipeline",level:2},{value:"\ud83c\udfdb\ufe0f Datasets",id:"\ufe0f-datasets",level:2},{value:"\ud83e\uddd0 Production Structure",id:"-production-structure",level:2},{value:"\u2696\ufe0f License",id:"\ufe0f-license",level:2}],p=e=>{let{children:t}=e;return(0,a.kt)("div",{style:{textAlign:"center"}},t)},u=e=>{let{headers:t,rows:n}=e;return(0,a.kt)("div",{style:{display:"inline-block"}},(0,a.kt)("table",null,(0,a.kt)("tbody",null,(0,a.kt)("tr",null,(0,a.kt)("td",{colSpan:3,style:{textAlign:"center"}},"Developed by four talented young men as their graduation project.")),(0,a.kt)("tr",null,t.map((e=>(0,a.kt)("th",{key:e},e)))),n.map(((e,t)=>(0,a.kt)("tr",{key:t},Array.isArray(e)?e.map(((e,t)=>(0,a.kt)("td",{key:t},e))):null))))))},d={toc:s,CenterContainer:p,Table:u},m="wrapper";function y(e){let{components:t,...n}=e;return(0,a.kt)(m,(0,r.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)(p,{mdxType:"CenterContainer"},(0,a.kt)("img",{src:"/img/ScanMate.png",alt:"ScanMate",width:"400"}),(0,a.kt)(u,{headers:["Name","Github","Twitter"],rows:[["Ziad Mansour",(0,a.kt)("a",{href:"https://github.com/ZiadMansourM"},"ZiadMansourM"),(0,a.kt)("a",{href:"https://twitter.com/Ziad_M_404"},"@ziad_m_404")],["Mohamed Wael","-","-"],["Maged Elosail","-","-"],["Yousif adel","-","-"]],mdxType:"Table"}),(0,a.kt)("p",null,"Developed by four talented young men as their graduation project.")),(0,a.kt)("h2",{id:"-pipeline"},"\ud83d\udcdd Pipeline"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-Console"},"*** We have the following 7 steps in our pipeline:\n$ prepare_images\n- Load Dataset Images\n- Compute Mask\n$ compute_sift_keypoints_descriptors\n$ image_matching\n$ data_feature_matching\n- Apply crossCheck BF Matcher\n- Apply Ransac on BF Matcher Output\n- Loop without repetition using Itertools\n$ compute_k_matrix\n$ generate_point_cloud\n- Recover Pose of reference camera\n- Recover rest camera poses using solvePNPRansac\n- Apply Triangulation\n$ 3D reconstruction\n- Use PointsCloud to generate a 3D Object (.stl) file\n")),(0,a.kt)("h2",{id:"\ufe0f-datasets"},"\ud83c\udfdb\ufe0f Datasets"),(0,a.kt)("ul",{className:"contains-task-list"},(0,a.kt)("li",{parentName:"ul",className:"task-list-item"},(0,a.kt)("input",{parentName:"li",type:"checkbox",checked:!1,disabled:!0})," ","snow-man."),(0,a.kt)("li",{parentName:"ul",className:"task-list-item"},(0,a.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","hammer."),(0,a.kt)("li",{parentName:"ul",className:"task-list-item"},(0,a.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","cottage."),(0,a.kt)("li",{parentName:"ul",className:"task-list-item"},(0,a.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","fountain.")),(0,a.kt)("h2",{id:"-production-structure"},"\ud83e\uddd0 Production Structure"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-console"},"(venv) ziadh@Ziads-MacBook-Air production % tree \n.\n\u251c\u2500\u2500 conf\n\u2502\xa0\xa0 \u251c\u2500\u2500 certs\n\u2502\xa0\xa0 \u251c\u2500\u2500 html\n\u2502\xa0\xa0 \u251c\u2500\u2500 kong-config\n\u2502\xa0\xa0 \u2502\xa0\xa0 \u2514\u2500\u2500 kong.yaml\n\u2502\xa0\xa0 \u251c\u2500\u2500 logs\n\u2502\xa0\xa0 \u2514\u2500\u2500 nginx.conf\n\u251c\u2500\u2500 data\n\u251c\u2500\u2500 docker-compose.yml\n\u2514\u2500\u2500 src\n    \u251c\u2500\u2500 Dockerfile\n    \u251c\u2500\u2500 main.py\n    \u251c\u2500\u2500 scanmate.py\n    \u2514\u2500\u2500 under_the_hood\n        \u251c\u2500\u2500 __init__.py\n        \u251c\u2500\u2500 compute_sift_features.py\n        \u251c\u2500\u2500 data_feature_match.py\n        \u251c\u2500\u2500 data_structures\n        \u2502\xa0\xa0 \u251c\u2500\u2500 __init__.py\n        \u2502\xa0\xa0 \u251c\u2500\u2500 feature_matches.py\n        \u2502\xa0\xa0 \u251c\u2500\u2500 image.py\n        \u2502\xa0\xa0 \u2514\u2500\u2500 images.py\n        \u251c\u2500\u2500 generate_points_cloud.py\n        \u251c\u2500\u2500 image_match.py\n        \u251c\u2500\u2500 prepare_images.py\n        \u2514\u2500\u2500 utils\n            \u251c\u2500\u2500 __init__.py\n            \u2514\u2500\u2500 utils.py\n\n10 directories, 18 files\n")),(0,a.kt)("h2",{id:"\ufe0f-license"},"\u2696\ufe0f License"),(0,a.kt)("p",null,"This project is licensed under the terms of the GNU General Public License version 3.0 (GPLv3). See the LICENSE file for details."))}y.isMDXComponent=!0}}]);