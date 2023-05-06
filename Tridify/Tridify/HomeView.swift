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
        Text(/*@START_MENU_TOKEN@*/"Hello, World!"/*@END_MENU_TOKEN@*/)
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}
