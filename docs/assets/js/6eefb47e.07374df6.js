"use strict";(self.webpackChunkscanmate=self.webpackChunkscanmate||[]).push([[779],{3905:(e,t,a)=>{a.d(t,{Zo:()=>l,kt:()=>u});var r=a(7294);function n(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){n(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,r,n=function(e,t){if(null==e)return{};var a,r,n={},i=Object.keys(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var p=r.createContext({}),m=function(e){var t=r.useContext(p),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},l=function(e){var t=m(e.components);return r.createElement(p.Provider,{value:t},e.children)},d="mdxType",c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},g=r.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,p=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),d=m(a),g=n,u=d["".concat(p,".").concat(g)]||d[g]||c[g]||i;return a?r.createElement(u,o(o({ref:t},l),{},{components:a})):r.createElement(u,o({ref:t},l))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,o=new Array(i);o[0]=g;var s={};for(var p in t)hasOwnProperty.call(t,p)&&(s[p]=t[p]);s.originalType=e,s[d]="string"==typeof e?e:n,o[1]=s;for(var m=2;m<i;m++)o[m]=a[m];return r.createElement.apply(null,o)}return r.createElement.apply(null,a)}g.displayName="MDXCreateElement"},8481:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>p,contentTitle:()=>o,default:()=>c,frontMatter:()=>i,metadata:()=>s,toc:()=>m});var r=a(7462),n=(a(7294),a(3905));const i={sidebar_position:2,id:"Prepare Images",description:"Load and mask image.",slug:"/under-the-hood/prepare-images"},o=void 0,s={unversionedId:"under-the-hood/Prepare Images",id:"under-the-hood/Prepare Images",title:"Prepare Images",description:"Load and mask image.",source:"@site/docs/under-the-hood/prepare_images.md",sourceDirName:"under-the-hood",slug:"/under-the-hood/prepare-images",permalink:"/under-the-hood/prepare-images",draft:!1,editUrl:"https://github.com/ZiadMansourM/photogrammetry/tree/main/docs/under-the-hood/prepare_images.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2,id:"Prepare Images",description:"Load and mask image.",slug:"/under-the-hood/prepare-images"},sidebar:"tutorialSidebar",previous:{title:"Introduction",permalink:"/under-the-hood/introduction"},next:{title:"Compute SIFT",permalink:"/under-the-hood/compute-sift"}},p={},m=[{value:"\ud83d\udcdd Prepare Images",id:"-prepare-images",level:2}],l={toc:m},d="wrapper";function c(e){let{components:t,...a}=e;return(0,n.kt)(d,(0,r.Z)({},l,a,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h2",{id:"-prepare-images"},"\ud83d\udcdd Prepare Images"),(0,n.kt)("ul",{className:"contains-task-list"},(0,n.kt)("li",{parentName:"ul",className:"task-list-item"},(0,n.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","Reads all images and their masks."),(0,n.kt)("li",{parentName:"ul",className:"task-list-item"},(0,n.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","Generate masks if they don't exist."),(0,n.kt)("li",{parentName:"ul",className:"task-list-item"},(0,n.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","Populate our ",(0,n.kt)("inlineCode",{parentName:"li"},"Images")," data structure.")),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-py"},'@timeit\ndef prepare_images(create_mask = True, **kwargs) -> Images:\n    image_set_name = kwargs[\'image_set_name\']\n    folder_path = f"../data/{image_set_name}"\n    images: Images = Images([], folder_path.split("/")[-1])\n    files: list[str] = list(\n        filter(\n            lambda file: ".jpg" in file, os.listdir(f"{folder_path}/images")\n        )\n    )\n    if create_mask:\n        for file in files:\n            image_path = f"{folder_path}/images/{file}"\n            rgb_image = OpenCV.cvtColor(OpenCV.imread(image_path), OpenCV.COLOR_BGR2RGB)\n            gray_image = OpenCV.cvtColor(rgb_image, OpenCV.COLOR_RGB2GRAY)\n            mask = remove(rgb_image)\n            mask = OpenCV.cvtColor(mask, OpenCV.COLOR_RGB2GRAY)\n            mask[mask > 0] = 255\n            OpenCV.imwrite(f"{folder_path}/masks/{file}", mask)\n            kernel = np.ones((5, 5), np.uint8)\n            dilated_mask = OpenCV.dilate(mask, kernel, iterations=20)\n            images.images.append(Image(file.split(".")[0], rgb_image, gray_image, dilated_mask, [], [], image_path))\n    else:\n        for file in files:\n            image_path = f"{folder_path}/images/{file}"\n            mask_path = f"{folder_path}/masks/{file}"\n            rgb_image = OpenCV.cvtColor(OpenCV.imread(image_path), OpenCV.COLOR_BGR2RGB)\n            gray_image = OpenCV.cvtColor(rgb_image, OpenCV.COLOR_RGB2GRAY)\n            mask = OpenCV.imread(mask_path, OpenCV.IMREAD_GRAYSCALE)\n            kernel = np.ones((5, 5), np.uint8)\n            dilated_mask = OpenCV.dilate(mask, kernel, iterations=20)\n            images.images.append(Image(file.split(".")[0], rgb_image, gray_image, dilated_mask, [], [], image_path))\n    return images\n')))}c.isMDXComponent=!0}}]);