//
//  HomeView.swift
//  Tridify
//
//  Created by Maged Alosali on 06/05/2023.
//

import SwiftUI

struct HomeView: View {
    
    @AppStorage("onBoarding") var onBoarding = false
    
    var body: some View {
        TabView {
            Text("First View")
                .tabItem {
                    Image(systemName: "scale.3d")
                    Text("Captures")
                }
            Text("Second View")
                .tabItem {
                    Image(systemName: "camera.fill")
                    Text("Camera")
                }
            
        }

    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
        HomeView().preferredColorScheme(.dark)
    }
}
